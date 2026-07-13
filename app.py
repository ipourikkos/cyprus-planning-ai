import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-terra")
QUERY_EXPANSION_MODEL = os.getenv("OPENAI_QUERY_EXPANSION_MODEL", ANSWER_MODEL)
VERIFY_MODEL = os.getenv("OPENAI_VERIFY_MODEL", ANSWER_MODEL)

SEMANTIC_THRESHOLD = 0.18
SEMANTIC_MATCH_COUNT_PER_QUERY = 16
FINAL_HITS = 40
LEXICAL_CANDIDATES = 40
ADJACENT_EXPANSION_TOP_N = 5
MAX_CONTEXT_CHARS = 26000
RERANK_CANDIDATES = 40
DIRECT_RULE_CANDIDATES = 12
RERANK_TOP_N = 10
RERANK_SNIPPET_CHARS = 1400
RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", ANSWER_MODEL)

SEMANTIC_WEIGHT = 0.64
LEXICAL_WEIGHT = 0.26
RECENCY_WEIGHT = 0.06
PRIORITY_WEIGHT = 0.04


GREEK_STOPWORDS = {
    "και", "ή", "η", "ο", "οι", "το", "τα", "του", "της", "των", "τον", "την",
    "σε", "στο", "στη", "στην", "στον", "στα", "στις", "στους", "με", "από",
    "για", "ως", "που", "ποιο", "ποια", "ποιος", "ποιες", "ποιοι", "τι", "πως",
    "πώς", "είναι", "ισχύει", "ισχυει", "ένα", "μια", "ένας", "αν", "να", "θα",
    "δεν", "τουλάχιστον", "μέχρι", "πάνω", "κάτω", "μεταξύ", "πρέπει",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "does", "do", "what", "how", "can", "must", "should", "from",
}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


def normalize_text(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return re.findall(r"[0-9a-zα-ω]+", normalized)


def meaningful_terms(question: str) -> List[str]:
    terms = []
    seen = set()

    for token in tokenize(question):
        if token in GREEK_STOPWORDS or token in ENGLISH_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)

    return terms


def token_root(token: str) -> str:
    """
    Light inflection-tolerant prefix matching.
    This is intentionally conservative: never shorter than 5 characters.
    """
    if len(token) >= 10:
        return token[:-3]
    if len(token) >= 7:
        return token[:-2]
    if len(token) >= 6:
        return token[:-1]
    return token


def parse_year(value: Any) -> int:
    if not value:
        return 0
    try:
        return int(str(value)[:4])
    except Exception:
        return 0


def recency_score(row: Dict[str, Any]) -> float:
    year = parse_year(row.get("publication_date"))
    if year >= 2025:
        return 1.0
    if year >= 2024:
        return 0.8
    if year >= 2020:
        return 0.5
    if year > 0:
        return 0.2
    return 0.0


def priority_score(row: Dict[str, Any]) -> float:
    try:
        return min(float(row.get("authority_priority") or 0.0), 100.0) / 100.0
    except Exception:
        return 0.0


def embed_text(text: str, openai_client: OpenAI) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def run_semantic_search(
    query_text: str,
    openai_client: OpenAI,
    supabase: Any,
) -> List[Dict[str, Any]]:
    query_embedding = embed_text(query_text, openai_client)

    response = supabase.rpc(
        "match_kb_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": SEMANTIC_THRESHOLD,
            "match_count": SEMANTIC_MATCH_COUNT_PER_QUERY,
        },
    ).execute()

    return response.data or []



def contains_greek(text: str) -> bool:
    return bool(re.search(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίϊΐόύϋΰώ]", text or ""))


def generate_greek_search_query(
    question: str,
    openai_client: OpenAI,
) -> str:
    """
    Convert a non-Greek user question into a concise Greek planning-regulation
    search query. This is for retrieval only, not for answering the user.
    """
    if contains_greek(question):
        return question

    instructions = """
You translate user questions into concise Greek search queries for a Cyprus
planning-regulations knowledge base.

Rules:
1. Do NOT answer the question.
2. Preserve the exact technical meaning.
3. Use Cyprus planning terminology where appropriate.
4. Prefer terms likely to appear in Greek planning documents, for example:
   - building coefficient -> συντελεστής δόμησης
   - coverage -> ποσοστό κάλυψης
   - basement -> υπόγειο
   - auxiliary building -> βοηθητική οικοδομή
   - setback / boundary distance -> απόσταση από τα σύνορα
   - parking space -> χώρος στάθμευσης
5. Return ONLY the Greek search query, with no quotation marks or explanation.
"""

    response = openai_client.responses.create(
        model=QUERY_EXPANSION_MODEL,
        instructions=instructions.strip(),
        input=question.strip(),
    )

    greek_query = (response.output_text or "").strip()

    if not greek_query:
        return question

    return greek_query


def merge_unique_rows(
    row_groups: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}

    for rows in row_groups:
        for row in rows:
            key = (
                row.get("document_id"),
                row.get("page_number"),
                row.get("content") or "",
            )

            existing = merged.get(key)
            if not existing:
                merged[key] = row
                continue

            # Keep the strongest values seen across original-language and
            # Greek-expanded retrieval runs.
            for field in (
                "similarity",
                "lexical_score",
                "raw_lexical_score",
                "direct_rule_score",
                "direct_score",
            ):
                new_value = float(row.get(field) or 0.0)
                old_value = float(existing.get(field) or 0.0)
                if new_value > old_value:
                    existing[field] = row.get(field)

    return list(merged.values())


def semantic_candidates(
    question: str,
    openai_client: OpenAI,
    supabase: Any,
) -> List[Dict[str, Any]]:
    queries = [
        question,
        (
            f"{question}\n"
            "Εξαιρέσεις, προϋποθέσεις, ειδικές περιπτώσεις, "
            "δεν προσμετράται, εξαιρείται, μερική προσμέτρηση, ανάλογα με τη χρήση."
        ),
        (
            f"{question}\n"
            "Ισχύουσες νεότερες πρόνοιες 2025, Εντολή 4/2024, "
            "τρέχοντες κανόνες και ειδικές εξαιρέσεις."
        ),
    ]

    merged: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}

    for query_text in queries:
        for row in run_semantic_search(query_text, openai_client, supabase):
            key = (
                row.get("document_id"),
                row.get("page_number"),
                row.get("content") or "",
            )
            existing = merged.get(key)
            if not existing or float(row.get("similarity") or 0.0) > float(existing.get("similarity") or 0.0):
                merged[key] = row

    return list(merged.values())


def fetch_all_chunks_with_metadata(supabase: Any) -> List[Dict[str, Any]]:
    docs_response = (
        supabase.table("kb_documents")
        .select("id,title,publisher,publication_date,version,authority_priority")
        .execute()
    )
    docs = {row["id"]: row for row in (docs_response.data or [])}

    chunks_response = (
        supabase.table("kb_chunks")
        .select("id,document_id,page_number,section_title,content")
        .execute()
    )

    rows = []
    for chunk in chunks_response.data or []:
        doc = docs.get(chunk.get("document_id"), {})
        rows.append({**chunk, **doc, "document_id": chunk.get("document_id")})

    return rows


def lexical_candidates(
    question: str,
    all_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    terms = meaningful_terms(question)
    if not terms:
        return []

    roots = {term: token_root(term) for term in terms}

    tokenized_rows = []
    document_frequency = Counter()

    for row in all_rows:
        section_text = row.get("section_title") or ""
        combined_text = f"{section_text}\n{row.get('content') or ''}"
        content_tokens = tokenize(combined_text)
        token_set = set(content_tokens)

        matched_terms = set()
        for term in terms:
            root = roots[term]
            if term in token_set or any(tok.startswith(root) for tok in token_set):
                matched_terms.add(term)

        for term in matched_terms:
            document_frequency[term] += 1

        tokenized_rows.append((row, content_tokens, token_set))

    total_docs = max(len(all_rows), 1)
    normalized_question = normalize_text(question)

    scored = []

    for row, content_tokens, token_set in tokenized_rows:
        score = 0.0
        exact_matches = 0
        root_matches = 0

        for term in terms:
            root = roots[term]
            df = document_frequency.get(term, 0)
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0

            if term in token_set:
                score += 1.0 * idf
                exact_matches += 1
            elif any(tok.startswith(root) for tok in token_set):
                score += 0.72 * idf
                root_matches += 1

        normalized_content = normalize_text(row.get("content") or "")
        normalized_section = normalize_text(row.get("section_title") or "")

        # Strong bonus when the query term appears in the detected section title.
        section_tokens = set(tokenize(row.get("section_title") or ""))
        section_match_count = 0
        for term in terms:
            root = roots[term]
            if term in section_tokens or any(tok.startswith(root) for tok in section_tokens):
                section_match_count += 1
                score += 3.5

        # Phrase bonus when several important question words occur near each other.
        matched_count = exact_matches + root_matches
        coverage = matched_count / max(len(terms), 1)
        score += coverage * 2.0

        if section_match_count:
            score += min(section_match_count, 3) * 1.5

        # Small exact-phrase bonus.
        if len(normalized_question) >= 8 and normalized_question in normalized_content:
            score += 4.0

        if score > 0:
            scored.append({**row, "raw_lexical_score": score})

    scored.sort(key=lambda r: float(r.get("raw_lexical_score") or 0.0), reverse=True)

    if not scored:
        return []

    max_score = float(scored[0]["raw_lexical_score"]) or 1.0
    for row in scored:
        row["lexical_score"] = float(row["raw_lexical_score"]) / max_score

    return scored[:LEXICAL_CANDIDATES]



DOMAIN_RELATION_EXPANSIONS = {
    # Questions like "Μετρά ... στον συντελεστή δόμησης;"
    "μετρ": [
        "υπολογισ", "λογιζ", "προσμετρ", "συνυπολογ", "εξαιρ",
    ],
    "λογιζ": [
        "υπολογισ", "μετρ", "προσμετρ", "συνυπολογ", "εξαιρ",
    ],
    "προσμετρ": [
        "υπολογισ", "λογιζ", "μετρ", "συνυπολογ", "εξαιρ",
    ],
    "εξαιρ": [
        "υπολογισ", "λογιζ", "μετρ", "προσμετρ", "συνυπολογ",
    ],
}


def rootify(token: str) -> str:
    token = normalize_text(token)
    if len(token) >= 10:
        return token[:-3]
    if len(token) >= 7:
        return token[:-2]
    if len(token) >= 6:
        return token[:-1]
    return token


def direct_rule_candidates(
    question: str,
    all_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    High-precision rule matching over the entire KB.

    This is designed to catch passages that literally encode the asked legal
    relationship, even when vector similarity or recency boosts rank them lower.

    Example:
    "Μετρά το υπόγειο στον συντελεστή δόμησης;"
    should strongly favor a chunk containing:
    "ΥΠΟΓΕΙΟ ... Εξαιρείται από τον υπολογισμό του συντελεστή δόμησης ..."
    """
    q_tokens = meaningful_terms(question)
    q_roots = [rootify(t) for t in q_tokens]

    # Expand relational verbs into legal-document wording.
    relation_roots = set()
    for root in q_roots:
        for trigger, expansions in DOMAIN_RELATION_EXPANSIONS.items():
            if root.startswith(trigger) or trigger.startswith(root):
                relation_roots.update(expansions)

    # Important concept roots are the non-stopword roots from the question.
    concept_roots = [r for r in q_roots if len(r) >= 4]

    scored = []

    for row in all_rows:
        combined = normalize_text(
            f"{row.get('section_title') or ''}\n{row.get('content') or ''}"
        )

        # Root-level concept matches.
        concept_hits = sum(1 for root in concept_roots if root in combined)
        relation_hits = sum(1 for root in relation_roots if root in combined)

        # Strong phrase/concept bonuses for planning-coefficient questions.
        building_coeff_bonus = 0.0
        if "συντελεστ" in combined and "δομησ" in combined:
            building_coeff_bonus = 4.0

        section_bonus = 0.0
        section_norm = normalize_text(row.get("section_title") or "")
        if any(root in section_norm for root in concept_roots):
            section_bonus = 3.0

        # Require at least meaningful concept overlap.
        if concept_hits == 0:
            continue

        score = (
            concept_hits * 2.2
            + relation_hits * 2.5
            + building_coeff_bonus
            + section_bonus
        )

        # Big bonus when multiple question concepts co-occur with a legal relation.
        if concept_hits >= 2 and relation_hits >= 1:
            score += 7.0
        if concept_hits >= 3 and relation_hits >= 1:
            score += 4.0

        # Direct exclusion/counting language is especially valuable.
        if "εξαιρ" in combined and "υπολογισ" in combined:
            score += 4.0
        if "λογιζ" in combined or "προσμετρ" in combined or "συνυπολογ" in combined:
            score += 2.0

        if score > 0:
            scored.append({**row, "direct_rule_score": score})

    scored.sort(
        key=lambda r: float(r.get("direct_rule_score") or 0.0),
        reverse=True,
    )

    return scored[:DIRECT_RULE_CANDIDATES]


def merge_and_rerank(
    semantic_rows: List[Dict[str, Any]],
    lexical_rows: List[Dict[str, Any]],
    direct_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}

    def key_for(row: Dict[str, Any]) -> Tuple[Any, Any, str]:
        return (
            row.get("document_id"),
            row.get("page_number"),
            row.get("content") or "",
        )

    for row in semantic_rows:
        key = key_for(row)
        merged[key] = {
            **row,
            "semantic_score": float(row.get("similarity") or 0.0),
            "lexical_score": 0.0,
        }

    for row in lexical_rows:
        key = key_for(row)
        if key in merged:
            merged[key]["lexical_score"] = float(row.get("lexical_score") or 0.0)
        else:
            merged[key] = {
                **row,
                "similarity": 0.0,
                "semantic_score": 0.0,
                "lexical_score": float(row.get("lexical_score") or 0.0),
            }

    # Force direct rule matches into the candidate pool.
    max_direct = max(
        [float(r.get("direct_rule_score") or 0.0) for r in direct_rows] or [1.0]
    )
    for row in direct_rows:
        key = key_for(row)
        normalized_direct = float(row.get("direct_rule_score") or 0.0) / max_direct
        if key in merged:
            merged[key]["direct_rule_score"] = float(row.get("direct_rule_score") or 0.0)
            merged[key]["direct_score"] = normalized_direct
        else:
            merged[key] = {
                **row,
                "similarity": 0.0,
                "semantic_score": 0.0,
                "lexical_score": 0.0,
                "direct_score": normalized_direct,
            }

    rows = list(merged.values())

    for row in rows:
        semantic = float(row.get("semantic_score") or 0.0)
        lexical = float(row.get("lexical_score") or 0.0)
        direct = float(row.get("direct_score") or 0.0)

        row["hybrid_score"] = (
            0.50 * semantic
            + 0.20 * lexical
            + 0.22 * direct
            + RECENCY_WEIGHT * recency_score(row)
            + PRIORITY_WEIGHT * priority_score(row)
        )

    rows.sort(
        key=lambda r: (
            float(r.get("hybrid_score") or 0.0),
            float(r.get("semantic_score") or 0.0),
            float(r.get("lexical_score") or 0.0),
        ),
        reverse=True,
    )

    return rows[:FINAL_HITS]



def llm_rerank_candidates(
    question: str,
    candidates: List[Dict[str, Any]],
    openai_client: OpenAI,
) -> List[Dict[str, Any]]:
    """
    Second-stage semantic/legal reranker.

    Hybrid retrieval is good at recall, but can still rank a merely related newer
    passage above an older passage that directly states the rule. This reranker
    sees the actual candidate text and prioritizes direct answerability first.
    """
    pool = candidates[:RERANK_CANDIDATES]
    if not pool:
        return []

    blocks = []
    for i, row in enumerate(pool, start=1):
        content = (row.get("content") or "").strip()
        if len(content) > RERANK_SNIPPET_CHARS:
            content = content[:RERANK_SNIPPET_CHARS] + "…"

        blocks.append(
            f"CANDIDATE {i}\n"
            f"Document: {row.get('title')}\n"
            f"Publication date: {row.get('publication_date')}\n"
            f"Page: {row.get('page_number')}\n"
            f"Section: {row.get('section_title') or 'Unknown'}\n"
            f"Hybrid score: {float(row.get('hybrid_score') or 0.0):.4f}\n"
            f"Text:\n{content}\n"
        )

    instructions = """
You rerank source excerpts for a Cyprus planning-regulations question.

Rank by DIRECT ANSWERABILITY first:
1. A passage that explicitly states the rule asked about ranks above a passage that is merely related.
2. A passage containing the exact legal relationship in the question ranks highly even if it is older.
3. Newer sources matter for current applicability, but do not bury an older passage that directly states the rule; include both when the newer source may qualify it.
4. Prefer passages containing conditions, exceptions, exclusions, and definitions that materially affect the answer.
5. Do not answer the user's question. Only rank candidate indices.

Return ONLY valid JSON in this exact shape:
{"ranked_indices":[1,2,3,4,5,6,7,8,9,10]}

Use at most 10 indices. Do not include indices that are not useful.
"""

    prompt = (
        f"QUESTION:\n{question}\n\n"
        "CANDIDATES:\n\n"
        + "\n\n".join(blocks)
    )

    try:
        response = openai_client.responses.create(
            model=RERANK_MODEL,
            instructions=instructions.strip(),
            input=prompt.strip(),
        )
        text = response.output_text.strip()

        # Be tolerant if the model accidentally wraps JSON in prose/code fences.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return pool[:RERANK_TOP_N]

        data = json.loads(text[start:end + 1])
        indices = data.get("ranked_indices", [])

        reranked = []
        seen = set()
        for idx in indices:
            try:
                pos = int(idx) - 1
            except Exception:
                continue
            if 0 <= pos < len(pool) and pos not in seen:
                reranked.append(pool[pos])
                seen.add(pos)
            if len(reranked) >= RERANK_TOP_N:
                break

        # Fill any remaining slots from the original hybrid order.
        for pos, row in enumerate(pool):
            if pos not in seen:
                reranked.append(row)
                seen.add(pos)
            if len(reranked) >= RERANK_TOP_N:
                break

        return reranked

    except Exception as exc:
        print(f"Reranker warning: {exc}")
        print("Falling back to hybrid ranking.")
        return pool[:RERANK_TOP_N]


def expand_with_adjacent_pages(
    rows: List[Dict[str, Any]],
    supabase: Any,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, Any, str]] = set()

    for row in rows:
        key = (
            row.get("document_id"),
            row.get("page_number"),
            row.get("content") or "",
        )
        if key not in seen:
            expanded.append({**row, "context_type": "hybrid_hit"})
            seen.add(key)

    for hit in rows[:ADJACENT_EXPANSION_TOP_N]:
        document_id = hit.get("document_id")
        page = hit.get("page_number")
        if not document_id or not page:
            continue

        start_page = max(1, int(page) - 1)
        end_page = int(page) + 1

        response = (
            supabase.table("kb_chunks")
            .select("document_id,page_number,section_title,content")
            .eq("document_id", document_id)
            .gte("page_number", start_page)
            .lte("page_number", end_page)
            .order("page_number")
            .execute()
        )

        for neighbor in response.data or []:
            key = (
                neighbor.get("document_id"),
                neighbor.get("page_number"),
                neighbor.get("content") or "",
            )
            if key in seen:
                continue

            expanded.append(
                {
                    **neighbor,
                    "title": hit.get("title"),
                    "publisher": hit.get("publisher"),
                    "publication_date": hit.get("publication_date"),
                    "version": hit.get("version"),
                    "authority_priority": hit.get("authority_priority"),
                    "similarity": None,
                    "semantic_score": None,
                    "lexical_score": None,
                    "hybrid_score": None,
                    "context_type": "adjacent_page_context",
                }
            )
            seen.add(key)

    return expanded


def build_context(rows: List[Dict[str, Any]]) -> str:
    primary = [r for r in rows if r.get("context_type") == "hybrid_hit"]
    adjacent = [r for r in rows if r.get("context_type") == "adjacent_page_context"]

    adjacent.sort(
        key=lambda r: (
            str(r.get("title") or ""),
            int(r.get("page_number") or 0),
        )
    )

    ordered = primary + adjacent

    blocks = []
    total_chars = 0

    for i, row in enumerate(ordered, start=1):
        title = row.get("title") or "Unknown document"
        page = row.get("page_number") or "?"
        pub_date = row.get("publication_date") or "unknown"
        priority = row.get("authority_priority") or 0
        context_type = row.get("context_type") or "source"
        section_title = row.get("section_title") or "Μη καθορισμένη"
        content = (row.get("content") or "").strip()

        block = (
            f"[SOURCE {i}]\n"
            f"Document: {title}\n"
            f"Page: {page}\n"
            f"Publication date: {pub_date}\n"
            f"Internal source priority: {priority}\n"
            f"Context type: {context_type}\n"
            f"Section: {section_title}\n"
            f"Text:\n{content}\n"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n".join(blocks)



def output_language_for_question(question: str) -> str:
    return "Greek" if contains_greek(question) else "English"


def answer_body_language_mismatch(text: str, target_language: str) -> bool:
    """
    Ignore bracketed citations because Greek document titles may legitimately
    appear inside an otherwise English answer.
    """
    body = re.sub(r"\[[^\]]+\]", " ", text or "")
    greek_letters = len(re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίϊΐόύϋΰώ]", body))
    latin_letters = len(re.findall(r"[A-Za-z]", body))

    if target_language == "English":
        return greek_letters > max(30, latin_letters * 0.35)

    return latin_letters > max(60, greek_letters * 0.80)


def answer_question(
    question: str,
    rows: List[Dict[str, Any]],
    openai_client: OpenAI,
) -> str:
    if not rows:
        return (
            "Δεν βρέθηκαν επαρκώς σχετικά αποσπάσματα στη βάση γνώσης. "
            "Δοκίμασε να διατυπώσεις διαφορετικά την ερώτηση."
        )

    context = build_context(rows)
    target_language = output_language_for_question(question)
    required_note = (
        "Σημείωση: Η απάντηση βασίζεται στα διαθέσιμα έγγραφα της βάσης γνώσης και δεν υποκαθιστά επίσημη νομική ή πολεοδομική γνωμάτευση."
        if target_language == "Greek"
        else "Note: This answer is based on the available documents in the knowledge base and does not replace official legal or planning advice."
    )

    instructions = f"""
You are a Cyprus planning-regulations research assistant for architects.

OUTPUT LANGUAGE: {target_language}
The source excerpts may be in Greek. Ignore the source language when choosing the output language.
You must write the answer in {target_language} because that is the language of the user's question.

You must answer ONLY from the supplied retrieved source excerpts.

SOURCE PRECEDENCE:
- First identify the newest directly applicable source.
- When a 2025 source and a 2011 source differ, do not silently follow the 2011 source.
- Use an older source only when it is consistent with newer material or when no newer applicable material is available.
- The internal source-priority number is only a retrieval hint, not a legal hierarchy.

LEGAL-READING RULES:
1. Never invent a regulation, number, exception, interpretation, or citation.
2. Never give a universal "yes" or "no" when the excerpts show that the answer depends on use, conditions, exceptions, discretion, or a category of space.
3. Before answering any yes/no question, explicitly check the supplied excerpts for:
   - exceptions
   - exclusions
   - partial counting
   - conditions
   - distinctions by use
   - newer rules that qualify older guidance
4. If the correct answer is conditional, start with "Εξαρτάται" in Greek or "It depends" in English.
5. Read neighboring page excerpts as continuous context across page breaks.
6. Resolve pronouns from preceding context before stating what an exception applies to.
7. Never generalize an exception from a specific object or use to a broader category.
8. NEVER combine conditions from separate provisions into one cumulative condition unless the source explicitly says they all apply together.
9. Treat the following as separate legal categories unless the source explicitly joins them:
   - general rule
   - definition
   - mandatory conditions
   - exception
   - special fire-safety provision
   - discretionary power of the Competent Authority
10. A special fire-safety rule must never be presented as a condition of the ordinary/general rule unless the source explicitly says so.
11. If the excerpts are insufficient or ambiguous, say so clearly.
12. Distinguish the general rule from exceptions and discretionary powers.
13. Answer in the same language as the user's question.
14. Be concise but practically useful to an architect.
15. Cite factual claims inline using:
    [Document title, p. X]
    or [Document title, pp. X–Y]
16. Every citation must include the FULL document title. Never shorten a citation to [p. X], [pp. X–Y], [σ. X], or [σσ. X–Y].
17. Do not cite SOURCE numbers.
18. End with exactly this note:
    {required_note}
"""

    prompt = f"""
USER QUESTION:
{question}

HYBRID-RETRIEVED SOURCE EXCERPTS:
{context}

Before drafting the answer, internally build a small legal rule map:
- GENERAL RULE
- DEFINITIONS
- MANDATORY CONDITIONS
- EXCEPTIONS
- SPECIAL CASES
- DISCRETIONARY POWERS
- SOURCE FOR EACH PROPOSITION

Do not show this internal map to the user.

Then:
- Identify the newest directly applicable source.
- Check exact keyword matches as well as semantic context.
- Check whether the answer has exceptions or depends on the type/use of space.
- Check whether any older source is qualified by newer material.
- Do not turn separate exceptions or special cases into extra conditions of the general rule.
- Do not join two source statements with "and", "provided that", or equivalent wording unless the source itself makes them cumulative.
- ALWAYS state the most directly applicable general rule first when the sources provide one.
- Do not replace an explicit general rule with a broad opening such as "it depends" or "there is no single rule".
- Put exceptions, limitations, unusual scenarios, and special zones after the general rule.
- If the question is broad, answer the ordinary/common case first, then explain when a different rule may apply.

Then write only the draft evidence-grounded answer.
"""

    response = openai_client.responses.create(
        model=ANSWER_MODEL,
        instructions=instructions.strip(),
        input=prompt.strip(),
    )

    draft_answer = response.output_text.strip()

    verifier_instructions = f"""
You are the final legal-consistency verifier for a Cyprus planning-regulations assistant.

You receive:
1. the user's question,
2. the exact retrieved source excerpts,
3. a draft answer.

OUTPUT LANGUAGE: {target_language}
Your job is to return a corrected final answer in {target_language}.
The source excerpts may be Greek. Do NOT switch to Greek merely because the source material is Greek.

Check especially for SYNTHESIS ERRORS:
- Did the draft combine separate provisions into one cumulative condition?
- Did it turn an exception into a condition of the general rule?
- Did it turn a special fire-safety provision into a general requirement?
- Did it generalize a discretionary power?
- Did it merge facts from different source passages using "and", "provided that", or similar wording when the sources do not make them cumulative?
- Did it state something stronger than the excerpts support?
- Are general rule, conditions, exceptions, special cases, and discretionary powers clearly separated?
- If the sources contain a directly applicable general rule, is it stated first?
- Did the draft incorrectly open with "it depends" or "there is no single rule" even though a general rule is available?
- Are citations attached to the claims they actually support?

Rules:
1. Correct any such error.
2. Preserve useful, accurate content.
3. Do not add facts that are not in the excerpts.
4. Write the prose in {target_language}.
5. Every citation must include the FULL document title, for example:
   [Document title, p. X] or [Document title, pp. X–Y]
   Never shorten citations to [p. X], [pp. X–Y], [σ. X], or [σσ. X–Y].
6. Preserve Greek document titles inside citations even when the answer is in English.
7. Do not mention that you reviewed or corrected a draft.
8. End with exactly this note:
   {required_note}
9. Return ONLY the final answer to the user.
"""

    verifier_prompt = f"""
TARGET OUTPUT LANGUAGE:
{target_language}

USER QUESTION:
{question}

SOURCE EXCERPTS:
{context}

DRAFT ANSWER:
{draft_answer}

Return the corrected final answer only.
"""

    verified = openai_client.responses.create(
        model=VERIFY_MODEL,
        instructions=verifier_instructions.strip(),
        input=verifier_prompt.strip(),
    )

    final_answer = verified.output_text.strip()

    # Deterministic safeguard: if the verifier still switches language because
    # the source excerpts are Greek, rewrite only the prose language while
    # preserving meaning and full citations.
    if answer_body_language_mismatch(final_answer, target_language):
        language_fix_instructions = f"""
Rewrite the supplied answer in {target_language}.

Rules:
1. Preserve the legal meaning exactly.
2. Do not add or remove substantive claims.
3. Preserve every citation and its full Greek document title.
4. Every citation must remain in the form [Document title, p. X] or [Document title, pp. X–Y].
5. Never use bare citations such as [p. X], [σ. X], or [σσ. X–Y].
6. End with exactly this note:
   {required_note}
7. Return ONLY the rewritten final answer.
"""
        language_fixed = openai_client.responses.create(
            model=VERIFY_MODEL,
            instructions=language_fix_instructions.strip(),
            input=final_answer,
        )
        final_answer = language_fixed.output_text.strip()

    return final_answer


def main() -> None:
    load_dotenv()

    supabase_url = require_env("SUPABASE_URL")
    supabase_secret_key = require_env("SUPABASE_SECRET_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")

    supabase = create_client(supabase_url, supabase_secret_key)
    openai_client = OpenAI(api_key=openai_api_key)

    print(f"Cyprus Planning AI v11 — model: {ANSWER_MODEL}")
    print("Bilingual retrieval + hybrid search + legal verification + output-language guard are ON.")
    print("General-rule-first answers + condition/exception separation + full citations are ON.")
    print("Type 'exit' to quit.\n")

    # Only 424 chunks currently, so loading all rows for local lexical scoring is cheap.
    all_rows = fetch_all_chunks_with_metadata(supabase)
    print(f"Loaded {len(all_rows)} knowledge-base chunks for lexical search.\n")

    while True:
        question = input("Ask a planning question:\n> ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        try:
            greek_search_query = generate_greek_search_query(question, openai_client)

            if greek_search_query != question:
                print(f"Greek retrieval query: {greek_search_query}")

            semantic_rows = merge_unique_rows([
                semantic_candidates(question, openai_client, supabase),
                semantic_candidates(greek_search_query, openai_client, supabase),
            ])

            lexical_rows = merge_unique_rows([
                lexical_candidates(question, all_rows),
                lexical_candidates(greek_search_query, all_rows),
            ])

            direct_rows = merge_unique_rows([
                direct_rule_candidates(question, all_rows),
                direct_rule_candidates(greek_search_query, all_rows),
            ])

            hybrid_rows = merge_and_rerank(semantic_rows, lexical_rows, direct_rows)
            reranked_rows = llm_rerank_candidates(question, hybrid_rows, openai_client)
            context_rows = expand_with_adjacent_pages(reranked_rows, supabase)

            print(
                f"\nSemantic candidates: {len(semantic_rows)} | "
                f"Lexical candidates: {len(lexical_rows)} | "
                f"Direct-rule candidates: {len(direct_rows)} | "
                f"Hybrid pool: {len(hybrid_rows)} | "
                f"LLM-reranked hits: {len(reranked_rows)} | "
                f"Context chunks: {len(context_rows)}"
            )
            print("Generating answer...\n")

            answer = answer_question(question, context_rows, openai_client)
            print(answer)

            print("\nTop LLM-reranked retrieval hits:")
            for i, row in enumerate(reranked_rows[:10], start=1):
                print(
                    f"{i}. {row.get('title')} — p. {row.get('page_number')} "
                    f"(semantic {float(row.get('semantic_score') or 0.0):.3f}, "
                    f"lexical {float(row.get('lexical_score') or 0.0):.3f}, "
                    f"direct {float(row.get('direct_score') or 0.0):.3f}, "
                    f"hybrid {float(row.get('hybrid_score') or 0.0):.3f})"
                )

            print("\n" + "=" * 90 + "\n")

        except Exception as exc:
            print(f"\nERROR: {exc}\n")
            print(
                "If this is a model-access error, set OPENAI_CHAT_MODEL in .env "
                "to a model available to your API project."
            )
            print()

# =========================
# WEB APP / API LAYER
# =========================

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


state: dict[str, Any] = {}


def unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int | None]] = set()
    sources: list[dict[str, Any]] = []

    for row in rows:
        title = row.get("title") or "Unknown document"
        page_number = row.get("page_number")
        key = (title, page_number)

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "title": title,
                "page_number": page_number,
                "section_title": row.get("section_title"),
                "publication_date": (
                    str(row.get("publication_date"))
                    if row.get("publication_date")
                    else None
                ),
                "version": row.get("version"),
                "publisher": row.get("publisher"),
            }
        )

    return sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    supabase_url = require_env("SUPABASE_URL")
    supabase_secret_key = require_env("SUPABASE_SECRET_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")

    state["supabase"] = create_client(supabase_url, supabase_secret_key)
    state["openai"] = OpenAI(api_key=openai_api_key)
    state["all_rows"] = fetch_all_chunks_with_metadata(state["supabase"])

    print(
        f"Cyprus Planning AI ready — loaded "
        f"{len(state['all_rows'])} knowledge-base chunks."
    )

    yield
    state.clear()


app = FastAPI(
    title="Cyprus Planning AI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chunks_loaded": len(state.get("all_rows", [])),
        "model": ANSWER_MODEL,
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = payload.question.strip()

    supabase = state.get("supabase")
    openai_client = state.get("openai")
    all_rows = state.get("all_rows")

    if not supabase or not openai_client or all_rows is None:
        raise HTTPException(status_code=503, detail="Service is not ready yet.")

    try:
        greek_search_query = generate_greek_search_query(question, openai_client)

        semantic_rows = merge_unique_rows([
            semantic_candidates(question, openai_client, supabase),
            semantic_candidates(greek_search_query, openai_client, supabase),
        ])

        lexical_rows = merge_unique_rows([
            lexical_candidates(question, all_rows),
            lexical_candidates(greek_search_query, all_rows),
        ])

        direct_rows = merge_unique_rows([
            direct_rule_candidates(question, all_rows),
            direct_rule_candidates(greek_search_query, all_rows),
        ])

        hybrid_rows = merge_and_rerank(
            semantic_rows,
            lexical_rows,
            direct_rows,
        )

        reranked_rows = llm_rerank_candidates(
            question,
            hybrid_rows,
            openai_client,
        )

        context_rows = expand_with_adjacent_pages(
            reranked_rows,
            supabase,
        )

        answer = answer_question(
            question,
            context_rows,
            openai_client,
        )

        return {
            "question": question,
            "answer": answer,
            "language": output_language_for_question(question),
            "greek_search_query": greek_search_query,
            "sources": unique_sources(reranked_rows),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Planning AI request failed: {exc}",
        ) from exc


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cyprus Planning AI</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f4f2;color:#17211b}
header{height:78px;padding:15px 26px;border-bottom:1px solid #dfe4e0;background:#fff;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0}
.eyebrow{font-size:11px;letter-spacing:.14em;font-weight:700;color:#68726c}.title{font-size:22px;font-weight:750;margin-top:4px}
.status{font-size:13px;color:#68726c}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#2f8f5b;margin-right:7px}
main{width:min(940px,100%);margin:auto;padding:42px 20px 180px}.welcome{text-align:center;max-width:700px;margin:40px auto}.icon{width:50px;height:50px;border-radius:14px;background:#173f2b;color:#fff;display:grid;place-items:center;margin:0 auto 18px;font-size:23px}
h1{font-size:31px;margin:0 0 10px}.muted{color:#68726c;line-height:1.6}.examples{display:grid;gap:10px;margin-top:26px}
.example{border:1px solid #dfe4e0;background:#fff;border-radius:14px;padding:14px 16px;text-align:left;cursor:pointer}
.row{display:flex;margin:22px 0}.user{justify-content:flex-end}.bubble{max-width:72%;background:#173f2b;color:#fff;border-radius:18px 18px 4px 18px;padding:13px 16px;line-height:1.5}
.card{width:100%;background:#fff;border:1px solid #dfe4e0;border-radius:18px;padding:22px;box-shadow:0 8px 28px rgba(21,44,30,.05)}
.label{color:#173f2b;font-size:11px;font-weight:800;letter-spacing:.12em;margin-bottom:14px}.answer{line-height:1.7;font-size:15.5px;white-space:pre-wrap}
details{margin-top:18px;border-top:1px solid #dfe4e0;padding-top:14px}summary{cursor:pointer;color:#68726c;font-weight:600}.source{background:#e9f0eb;border:1px solid #d7e2da;border-radius:12px;padding:12px 14px;margin-top:10px}.source-title{font-weight:700}.source-meta{font-size:12px;color:#68726c;margin-top:5px}
.composer-wrap{position:fixed;left:0;right:0;bottom:0;padding:18px 20px 14px;background:linear-gradient(to top,#f3f4f2 72%,rgba(243,244,242,0))}
form{width:min(900px,calc(100% - 36px));margin:auto;background:#fff;border:1px solid #cfd8d1;border-radius:18px;padding:10px 10px 10px 16px;display:flex;gap:10px;align-items:flex-end;box-shadow:0 12px 34px rgba(20,42,28,.1)}
textarea{flex:1;border:0;resize:none;outline:none;min-height:42px;max-height:180px;padding:10px 2px;font:inherit;line-height:1.45}
button.send{border:0;background:#173f2b;color:#fff;border-radius:12px;padding:11px 18px;font-weight:700;cursor:pointer}button:disabled{opacity:.55}
.note{width:min(900px,calc(100% - 36px));margin:8px auto 0;text-align:center;font-size:11px;color:#68726c}
.error{color:#9b2c2c}
@media(max-width:700px){header{padding:14px 16px}.bubble{max-width:88%}.card{padding:17px}}
</style>
</head>
<body>
<header>
  <div><div class="eyebrow">CYPRUS PLANNING INTELLIGENCE</div><div class="title">Cyprus Planning AI</div></div>
  <div class="status"><span class="dot"></span><span id="statusText">Checking…</span></div>
</header>

<main id="messages">
  <section class="welcome" id="welcome">
    <div class="icon">⌂</div>
    <h1>Ask a planning question</h1>
    <p class="muted">Ask in English or Greek. Answers are grounded in the planning documents loaded in the knowledge base.</p>
    <div class="examples">
      <button class="example">Does a basement count toward the building coefficient?</button>
      <button class="example">How many parking spaces are required for a house?</button>
      <button class="example">Πώς μετριέται το ύψος σε επικλινές έδαφος;</button>
    </div>
  </section>
</main>

<div class="composer-wrap">
  <form id="form">
    <textarea id="input" rows="1" placeholder="Ask about Cyprus planning regulations…"></textarea>
    <button class="send" id="send" type="submit">Ask</button>
  </form>
  <div class="note">Research assistant only. Verify critical decisions against the official applicable planning instruments.</div>
</div>

<script>
const messages=document.getElementById("messages");
const form=document.getElementById("form");
const input=document.getElementById("input");
const send=document.getElementById("send");

function addUser(text){
  const row=document.createElement("div");row.className="row user";
  const bubble=document.createElement("div");bubble.className="bubble";bubble.textContent=text;
  row.appendChild(bubble);messages.appendChild(row);
}

function addLoading(){
  const row=document.createElement("div");row.className="row";
  row.innerHTML='<div class="card"><div class="label">PLANNING AI</div><div class="answer">Searching planning sources and checking the answer…</div></div>';
  messages.appendChild(row);return row;
}

function addAssistant(data){
  const row=document.createElement("div");row.className="row";
  const card=document.createElement("div");card.className="card";
  const label=document.createElement("div");label.className="label";label.textContent="PLANNING AI";
  const answer=document.createElement("div");answer.className="answer";answer.textContent=data.answer;
  card.append(label,answer);

  if(data.sources && data.sources.length){
    const details=document.createElement("details");
    const summary=document.createElement("summary");summary.textContent="Sources used";
    details.appendChild(summary);
    const seen=new Set();
    data.sources.forEach(s=>{
      const key=s.title+"|"+s.page_number;if(seen.has(key))return;seen.add(key);
      const box=document.createElement("div");box.className="source";
      const t=document.createElement("div");t.className="source-title";t.textContent=s.title;
      const m=document.createElement("div");m.className="source-meta";
      const parts=[];if(s.page_number!=null)parts.push("PDF page "+s.page_number);if(s.section_title)parts.push(s.section_title);if(s.publication_date)parts.push(s.publication_date);
      m.textContent=parts.join(" · ");box.append(t,m);details.appendChild(box);
    });
    card.appendChild(details);
  }
  row.appendChild(card);messages.appendChild(row);
}

async function ask(q){
  q=q.trim();if(!q)return;
  const welcome=document.getElementById("welcome");if(welcome)welcome.remove();
  addUser(q);input.value="";send.disabled=true;
  const loading=addLoading();window.scrollTo({top:document.body.scrollHeight,behavior:"smooth"});
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
    const data=await r.json();loading.remove();
    if(!r.ok)throw new Error(data.detail||"Request failed");
    addAssistant(data);
  }catch(e){
    loading.remove();
    const row=document.createElement("div");row.className="row";
    row.innerHTML='<div class="card error">Could not get an answer: '+String(e.message)+'</div>';
    messages.appendChild(row);
  }finally{
    send.disabled=false;input.focus();window.scrollTo({top:document.body.scrollHeight,behavior:"smooth"});
  }
}

form.addEventListener("submit",e=>{e.preventDefault();ask(input.value)});
input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
document.querySelectorAll(".example").forEach(b=>b.addEventListener("click",()=>ask(b.textContent)));

fetch("/health").then(r=>r.json()).then(d=>document.getElementById("statusText").textContent="Online · "+d.chunks_loaded+" chunks").catch(()=>document.getElementById("statusText").textContent="Offline");
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def homepage() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)
