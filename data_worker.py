"""PLANA.CY Data Worker V5

One persistent Render Background Worker for:
- BuySell sale listings
- BuySell rental listings
- DLS supporting ArcGIS layers
- Official market datasets: DLS statistics, CBC RPPI, CYSTAT construction costs

The parcel indexer intentionally stays separate.

Required environment variables:
  SUPABASE_URL
  SUPABASE_SECRET_KEY

Recommended optional variables:
  PLANA_DATA_CYCLE_SECONDS=900
  PLANA_MARKET_MAX_PAGE=500
  PLANA_MARKET_PAGES_PER_CYCLE=2
  PLANA_MARKET_DETAIL_DELAY=2.5
  PLANA_MARKET_BLOCK_BACKOFF_SECONDS=1800
  PLANA_DLS_LAYERS=11,12,13,15,16,17,18,19,28,31,32,35,36,37
  PLANA_DLS_BATCH=75
  PLANA_OFFICIAL_EVERY_CYCLES=24
  PLANA_DLS_EVERY_CYCLES=8
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader
from shapely.geometry import shape
from shapely import force_2d
from supabase import create_client

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BUYSELL_BASE = "https://www.buysellcyprus.com"
DLS_ARCGIS_BASE = "https://eservices.dls.moi.gov.cy/arcgis/rest/services/National/CadastralMap_EN/MapServer"
CBC_RPPI_PAGE = "https://www.centralbank.cy/en/publications/residential-property-price-indices"
CYSTAT_COST_PAGE = "https://www.cystat.gov.cy/en/KeyFiguresList?p=0&s=31&tID=3"

DLS_STATS = {
    "transfers": (
        "DLS Transfers of Sales",
        "https://portal.dls.moi.gov.cy/en/stats_category/enimerosi/statistika/poliseon/",
    ),
    "contracts": (
        "DLS Contracts of Sales",
        "https://portal.dls.moi.gov.cy/en/stats_category/enimerosi/statistika/politirion-engrafon/",
    ),
    "foreign_buyers": (
        "DLS Foreign Buyers",
        "https://portal.dls.moi.gov.cy/en/stats_category/enimerosi/statistika/poliseon-se-allodapous/",
    ),
    "mortgages": (
        "DLS Mortgages",
        "https://portal.dls.moi.gov.cy/en/stats_category/enimerosi/statistika/ypothikon/",
    ),
}

DLS_LAYERS = {
    11: "Development Plans",
    12: "Planning Zones",
    13: "Postal Code Areas",
    15: "Districts",
    16: "Municipalities Communities",
    17: "Quarters",
    18: "Blocks",
    19: "Localities",
    28: "Buildings",
    31: "Coast Protection Zone",
    32: "State Land",
    35: "Sporadic Survey Parcels",
    36: "Surveyed Parcels",
    37: "White Zones",
}

ID_RE = re.compile(r"-(\d+)\.html(?:$|\?)")
MONEY_RE = re.compile(r"€\s*([\d,.]+)")
EXTS = (".xlsx", ".xls", ".csv", ".pdf")

DATA_CYCLE_SECONDS = max(60, int(os.getenv("PLANA_DATA_CYCLE_SECONDS", "900")))
MARKET_MAX_PAGE = max(1, int(os.getenv("PLANA_MARKET_MAX_PAGE", "500")))
MARKET_PAGES_PER_CYCLE = max(1, int(os.getenv("PLANA_MARKET_PAGES_PER_CYCLE", "2")))
MARKET_DETAIL_DELAY = max(1.0, float(os.getenv("PLANA_MARKET_DETAIL_DELAY", "2.5")))
MARKET_BLOCK_BACKOFF_SECONDS = max(300, int(os.getenv("PLANA_MARKET_BLOCK_BACKOFF_SECONDS", "1800")))
DLS_BATCH = max(25, min(200, int(os.getenv("PLANA_DLS_BATCH", "75"))))
DLS_EVERY_CYCLES = max(1, int(os.getenv("PLANA_DLS_EVERY_CYCLES", "8")))
OFFICIAL_EVERY_CYCLES = max(1, int(os.getenv("PLANA_OFFICIAL_EVERY_CYCLES", "24")))

_raw_layers = os.getenv("PLANA_DLS_LAYERS", "11,12,13,15,16,17,18,19,28,31,32,35,36,37")
ACTIVE_DLS_LAYERS = [int(x.strip()) for x in _raw_layers.split(",") if x.strip() and int(x.strip()) in DLS_LAYERS]

UA = "PLANA.CY public market and geodata research collector/5.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def number(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"[\d,.]+", str(value))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def clean_cell(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


# -----------------------------------------------------------------------------
# Persistent checkpoint state
# -----------------------------------------------------------------------------


def state_get(sb, pipeline: str) -> dict[str, Any]:
    try:
        rows = (
            sb.table("plana_data_worker_state")
            .select("*")
            .eq("pipeline", pipeline)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else {}
    except Exception as exc:
        log(f"STATE {pipeline}: read failed, using defaults: {exc}")
        return {}


def state_put(sb, pipeline: str, **values: Any) -> None:
    payload = {"pipeline": pipeline, "updated_at": now(), **values}
    try:
        sb.table("plana_data_worker_state").upsert(payload, on_conflict="pipeline").execute()
    except Exception as exc:
        log(f"STATE {pipeline}: write failed: {exc}")


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------


async def async_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int = 4,
    skip_statuses: Iterable[int] = (),
) -> httpx.Response | None:
    delay = 3.0
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url)
            if response.status_code in set(skip_statuses):
                log(f"  blocked/unavailable status={response.status_code} url={url}")
                return None
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            log(f"  request failed {attempt}/{retries} status={status or 'network'} url={url}")
            if attempt < retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
    if last_error:
        log(f"  giving up url={url}: {last_error}")
    return None


def sync_get(client: httpx.Client, url: str, retries: int = 5) -> httpx.Response | None:
    delay = 4.0
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url)
            if response.status_code == 429:
                wait = min(20 * attempt, 120)
                log(f"  rate limited 429; sleeping {wait}s: {url}")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            log(f"  official request failed {attempt}/{retries} status={status or 'network'} url={url}")
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2, 40)
    return None


# -----------------------------------------------------------------------------
# BuySell sales + rentals
# -----------------------------------------------------------------------------


def listing_id(url: str) -> str | None:
    match = ID_RE.search(url)
    return match.group(1) if match else None


def walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def jsonlds(soup: BeautifulSoup) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            output.extend(x for x in walk_json(json.loads(tag.string or tag.get_text())) if isinstance(x, dict))
        except Exception:
            pass
    return output


def meta(soup: BeautifulSoup, key: str) -> str | None:
    tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
    return clean_text(tag.get("content")) if tag else None


def classify(title: str, url: str) -> str | None:
    text = f"{title} {url}".lower()
    checks = [
        ("apartment", ("apartment", "flat", "penthouse", "studio")),
        ("house", ("house", "villa", "bungalow", "maisonette", "townhouse")),
        ("land", ("land", "plot", "field")),
        ("office", ("office",)),
        ("shop", ("shop", "retail")),
        ("building", ("building", "block")),
        ("warehouse", ("warehouse", "industrial")),
    ]
    for kind, terms in checks:
        if any(term in text for term in terms):
            return kind
    return None


def parse_location(url: str, listing_status: str) -> tuple[str | None, str | None]:
    parts = [x for x in urlparse(url).path.split("/") if x]
    expected = f"property-for-{listing_status}"
    if len(parts) >= 4 and parts[0] == expected:
        return parts[1].replace("-", " ").title(), parts[2].replace("-", " ").title()
    return None, None


def links_from_search(html: str, listing_status: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen: dict[str, str] = {}
    needle = f"/property-for-{listing_status}/"
    for anchor in soup.find_all("a", href=True):
        url = urljoin(BUYSELL_BASE, anchor["href"]).split("#")[0]
        lid = listing_id(url)
        if lid and needle in url:
            seen[lid] = url
    return list(seen.values())


def parse_market_detail(url: str, html: str, listing_status: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    lid = listing_id(url)
    title = meta(soup, "og:title") or clean_text(soup.title.string if soup.title else None)
    description = meta(soup, "og:description")
    district, locality = parse_location(url, listing_status)
    row: dict[str, Any] = {
        "source": "buysell",
        "source_listing_id": lid,
        "url": url,
        "status": "active",
        "listing_status": listing_status,
        "title": title,
        "district": district,
        "locality": locality,
        "property_type": classify(title or "", url),
        "description": description,
        "last_seen_at": now(),
        "raw_data": {"parser": "data_worker_v5", "listing_status": listing_status},
    }

    raw_text = clean_text(soup.get_text(" ")) or ""
    for data in jsonlds(soup):
        offers = data.get("offers")
        if isinstance(offers, dict) and row.get("price_eur") is None:
            row["price_eur"] = number(offers.get("price"))
        if row.get("price_eur") is None and data.get("priceCurrency") == "EUR":
            row["price_eur"] = number(data.get("price"))
        address = data.get("address")
        if isinstance(address, dict):
            row["locality"] = clean_text(address.get("addressLocality")) or row["locality"]
            row["district"] = clean_text(address.get("addressRegion")) or row["district"]
        geo = data.get("geo")
        if isinstance(geo, dict):
            row["latitude"] = number(geo.get("latitude"))
            row["longitude"] = number(geo.get("longitude"))
        seller = data.get("seller") or data.get("provider")
        if isinstance(seller, dict) and not row.get("provider_name"):
            row["provider_name"] = clean_text(seller.get("name"))
        if not row.get("description"):
            row["description"] = clean_text(data.get("description"))

    if row.get("price_eur") is None:
        match = MONEY_RE.search(raw_text)
        if match:
            row["price_eur"] = number(match.group(1))

    patterns = {
        "internal_area_m2": [r"(?:internal|covered|living)\s+area\s*:?\s*([\d,.]+)\s*(?:m²|m2|sqm)"],
        "plot_area_m2": [r"(?:plot|land)\s+(?:area|size)\s*:?\s*([\d,.]+)\s*(?:m²|m2|sqm)"],
        "bedrooms": [r"bedrooms?\s*:?\s*(\d+)"],
        "bathrooms": [r"bathrooms?\s*:?\s*(\d+)"],
    }
    lower = raw_text.lower()
    for field, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, lower, re.I)
            if match:
                row[field] = number(match.group(1))
                break

    if row.get("latitude") is not None and row.get("longitude") is not None:
        row["geom"] = f"POINT({row['longitude']} {row['latitude']})"
    return row


def save_market_listing(sb, row: dict[str, Any]) -> None:
    existing = (
        sb.table("market_listings")
        .select("price_eur,status")
        .eq("source", row["source"])
        .eq("source_listing_id", row["source_listing_id"])
        .limit(1)
        .execute()
        .data
    )
    changed = (
        not existing
        or existing[0].get("price_eur") != row.get("price_eur")
        or existing[0].get("status") != "active"
    )
    sb.table("market_listings").upsert(row, on_conflict="source,source_listing_id").execute()
    if changed:
        sb.table("market_listing_history").insert(
            {
                "source": row["source"],
                "source_listing_id": row["source_listing_id"],
                "price_eur": row.get("price_eur"),
                "status": "active",
                "raw_data": {
                    "url": row["url"],
                    "title": row.get("title"),
                    "listing_status": row.get("listing_status"),
                },
            }
        ).execute()


async def market_pipeline(sb, listing_status: str) -> None:
    pipeline = f"market_{listing_status}"
    state = state_get(sb, pipeline)
    blocked_until = float(state.get("blocked_until_epoch") or 0)
    if blocked_until > time.time():
        wait = int(blocked_until - time.time())
        log(f"MARKET {listing_status}: checkpoint blocked for another {wait}s")
        return

    page = max(1, int(state.get("cursor_page") or 1))
    pages_done = 0
    seen_total = 0
    written_total = 0
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }
    limits = httpx.Limits(max_connections=2, max_keepalive_connections=1)

    async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=20), follow_redirects=True, headers=headers, limits=limits) as client:
        while pages_done < MARKET_PAGES_PER_CYCLE:
            search_url = f"{BUYSELL_BASE}/properties-for-{listing_status}/sort-rl/page-{page}"
            response = await async_get(client, search_url, retries=3, skip_statuses=(403, 429, 520, 521, 522, 523, 524))
            if response is None:
                until = time.time() + MARKET_BLOCK_BACKOFF_SECONDS
                state_put(
                    sb,
                    pipeline,
                    status="blocked",
                    cursor_page=page,
                    blocked_until_epoch=until,
                    last_error=f"Search page unavailable: {search_url}",
                )
                log(f"MARKET {listing_status}: page {page} blocked; preserving checkpoint and backing off")
                return

            links = links_from_search(response.text, listing_status)
            log(f"MARKET {listing_status}: page {page} links={len(links)}")
            if not links:
                page = 1
                state_put(sb, pipeline, status="wrapped", cursor_page=page, last_error=None)
                log(f"MARKET {listing_status}: no links; wrapping checkpoint to page 1")
                return

            for url in links:
                seen_total += 1
                detail = await async_get(client, url, retries=2, skip_statuses=(403, 429, 520, 521, 522, 523, 524))
                if detail is None:
                    await asyncio.sleep(MARKET_DETAIL_DELAY)
                    continue
                try:
                    row = parse_market_detail(url, detail.text, listing_status)
                    if row.get("source_listing_id"):
                        save_market_listing(sb, row)
                        written_total += 1
                except Exception as exc:
                    log(f"  MARKET {listing_status}: parse/save skipped {url}: {type(exc).__name__}: {exc}")
                await asyncio.sleep(MARKET_DETAIL_DELAY)

            page += 1
            if page > MARKET_MAX_PAGE:
                page = 1
            pages_done += 1
            state_put(
                sb,
                pipeline,
                status="running",
                cursor_page=page,
                pages_scanned=pages_done,
                items_seen=seen_total,
                items_written=written_total,
                last_error=None,
            )
            await asyncio.sleep(5)

    state_put(
        sb,
        pipeline,
        status="done",
        cursor_page=page,
        pages_scanned=pages_done,
        items_seen=seen_total,
        items_written=written_total,
        last_completed_at=now(),
        last_error=None,
    )
    log(f"MARKET {listing_status}: cycle pages={pages_done} seen={seen_total} written={written_total} next_page={page}")


# -----------------------------------------------------------------------------
# DLS ArcGIS supporting layers
# -----------------------------------------------------------------------------


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def geom_wkt(geometry: dict[str, Any] | None) -> str | None:
    if not geometry:
        return None
    geom = force_2d(shape(geometry))
    return None if geom.is_empty else geom.wkt


async def dls_json(client: httpx.AsyncClient, url: str, params: dict[str, Any], *, post: bool = False, retries: int = 5) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            response = await (client.post(url, data=params) if post else client.get(url, params=params))
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            return data
        except (httpx.RequestError, httpx.HTTPStatusError, RuntimeError) as exc:
            if attempt == retries:
                raise
            log(f"  DLS retry {attempt}/{retries} {url}: {type(exc).__name__}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"DLS request failed: {url}")


async def dls_layer_pipeline(sb, layer_id: int) -> None:
    name = DLS_LAYERS[layer_id]
    pipeline = f"dls_layer_{layer_id}"
    state = state_get(sb, pipeline)
    url = f"{DLS_ARCGIS_BASE}/{layer_id}"
    query_url = f"{url}/query"
    limits = httpx.Limits(max_connections=2, max_keepalive_connections=1)

    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=30), headers={"User-Agent": UA}, limits=limits) as client:
        try:
            meta = await dls_json(client, url, {"f": "json"})
            oid_field = meta.get("objectIdField") or next(
                (field["name"] for field in meta.get("fields", []) if field.get("type") == "esriFieldTypeOID"),
                None,
            )
            if not oid_field:
                raise RuntimeError("No object ID field")
            ids_data = await dls_json(client, query_url, {"f": "json", "where": "1=1", "returnIdsOnly": "true"})
            ids = sorted(set(ids_data.get("objectIds") or []))
            offset = max(0, int(state.get("cursor_offset") or 0))
            if offset >= len(ids):
                offset = 0
            chunk = ids[offset : offset + DLS_BATCH]
            if not chunk:
                state_put(sb, pipeline, status="done", cursor_offset=0, last_completed_at=now(), last_error=None)
                return

            data = await dls_json(
                client,
                query_url,
                {
                    "f": "geojson",
                    "objectIds": ",".join(map(str, chunk)),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": "4326",
                },
                post=True,
            )
            edit = (meta.get("editingInfo") or {}).get("lastEditDate")
            rows: list[dict[str, Any]] = []
            for feature in data.get("features") or []:
                props = {str(key): scalar(value) for key, value in (feature.get("properties") or {}).items()}
                object_id = props.get(oid_field)
                if object_id is None:
                    continue
                rows.append(
                    {
                        "layer_id": layer_id,
                        "layer_name": name,
                        "source_object_id": int(object_id),
                        "geom": geom_wkt(feature.get("geometry")),
                        "properties": props,
                        "source_url": url,
                        "source_last_edit_ms": edit,
                        "synced_at": now(),
                    }
                )

            for start in range(0, len(rows), 50):
                sb.table("dls_arcgis_features").upsert(rows[start : start + 50], on_conflict="layer_id,source_object_id").execute()

            next_offset = offset + len(chunk)
            complete = next_offset >= len(ids)
            state_put(
                sb,
                pipeline,
                status="done" if complete else "running",
                cursor_offset=0 if complete else next_offset,
                items_seen=len(chunk),
                items_written=len(rows),
                total_items=len(ids),
                last_completed_at=now() if complete else None,
                last_error=None,
            )
            sb.table("dls_sync_state").upsert(
                {
                    "layer_id": layer_id,
                    "layer_name": name,
                    "source_url": url,
                    "object_id_field": oid_field,
                    "geometry_type": meta.get("geometryType"),
                    "source_last_edit_ms": edit,
                    "feature_count": next_offset if not complete else len(ids),
                    "last_status": "done" if complete else "running",
                    "last_error": None,
                    "last_started_at": state.get("last_started_at") or now(),
                    "last_completed_at": now() if complete else None,
                    "updated_at": now(),
                },
                on_conflict="layer_id",
            ).execute()
            log(f"DLS layer {layer_id} {name}: {offset:,}->{next_offset:,}/{len(ids):,}; wrote={len(rows)}")
            del rows, data, ids
            gc.collect()
        except Exception as exc:
            state_put(sb, pipeline, status="error", last_error=f"{type(exc).__name__}: {exc}")
            try:
                sb.table("dls_sync_state").upsert(
                    {
                        "layer_id": layer_id,
                        "layer_name": name,
                        "source_url": url,
                        "last_status": "error",
                        "last_error": str(exc)[:2000],
                        "updated_at": now(),
                    },
                    on_conflict="layer_id",
                ).execute()
            except Exception:
                pass
            log(f"DLS layer {layer_id} failed; continuing: {type(exc).__name__}: {exc}")


async def dls_pipeline(sb) -> None:
    for layer_id in ACTIVE_DLS_LAYERS:
        await dls_layer_pipeline(sb, layer_id)
        await asyncio.sleep(1)
    for rpc_name in ("refresh_plana_dls_enrichment", "refresh_plana_dls_v2"):
        try:
            result = sb.rpc(rpc_name, {"p_limit": 100000}).execute().data
            log(f"DLS enrichment {rpc_name}: {result}")
        except Exception as exc:
            log(f"DLS enrichment {rpc_name} skipped: {exc}")


# -----------------------------------------------------------------------------
# Official market datasets
# -----------------------------------------------------------------------------


def dataset_key_for(prefix: str, url: str) -> str:
    filename = urlparse(url).path.rsplit("/", 1)[-1] or "download"
    filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename)[:150]
    return f"{prefix}:{filename}"


def discover_downloads(client: httpx.Client, page_url: str, *, follow_detail_pages: bool = True) -> list[str]:
    response = sync_get(client, page_url)
    if response is None:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    direct: list[str] = []
    details: list[str] = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(page_url, anchor["href"])
        label = " ".join(anchor.stripped_strings).lower()
        path = url.lower().split("?", 1)[0]
        if path.endswith(EXTS) or "download the file" in label or "data series" in label:
            direct.append(url)
        elif follow_detail_pages and (
            re.search(r"\b20(?:2[3-9]|[3-9]\d)\b", label)
            or "/statistics/" in url.lower()
        ):
            details.append(url)

    if follow_detail_pages:
        for detail_url in list(dict.fromkeys(details))[:30]:
            detail = sync_get(client, detail_url, retries=3)
            if detail is None:
                continue
            detail_soup = BeautifulSoup(detail.text, "html.parser")
            for anchor in detail_soup.find_all("a", href=True):
                url = urljoin(detail_url, anchor["href"])
                label = " ".join(anchor.stripped_strings).lower()
                path = url.lower().split("?", 1)[0]
                if path.endswith(EXTS) or "download the file" in label or "data series" in label:
                    direct.append(url)
            time.sleep(0.5)
    return list(dict.fromkeys(direct))


def parse_tabular(content: bytes, url: str, content_type: str) -> dict[str, pd.DataFrame]:
    lower = url.lower().split("?", 1)[0]
    content_type = content_type.lower()
    if lower.endswith(".csv") or "csv" in content_type:
        return {"csv": pd.read_csv(io.BytesIO(content), header=None)}
    # xlrd handles .xls, openpyxl handles .xlsx through pandas.
    return pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)


def parse_pdf_rows(content: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(content))
    rows: list[dict[str, Any]] = []
    row_number = 0
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = clean_text(line)
            if not line:
                continue
            row_number += 1
            rows.append(
                {
                    "sheet_name": f"page_{page_index}",
                    "row_number": row_number,
                    "row_data": {"text": line, "page": page_index},
                }
            )
    return rows


def write_official_dataset(
    sb,
    *,
    source: str,
    prefix: str,
    name: str,
    source_page: str,
    url: str,
    content: bytes,
    content_type: str,
) -> tuple[int, int]:
    dataset_key = dataset_key_for(prefix, url)
    sha = hashlib.sha256(content).hexdigest()
    lower = url.lower().split("?", 1)[0]
    is_pdf = lower.endswith(".pdf") or "application/pdf" in content_type.lower() or content[:4] == b"%PDF"

    rows_payload: list[dict[str, Any]] = []
    sheet_names: list[str] = []
    parser = "pdf_text" if is_pdf else "tabular"
    if is_pdf:
        for row in parse_pdf_rows(content):
            rows_payload.append(
                {
                    "source": source,
                    "dataset_key": dataset_key,
                    "sheet_name": row["sheet_name"],
                    "row_number": row["row_number"],
                    "row_data": row["row_data"],
                    "synced_at": now(),
                }
            )
        sheet_names = sorted({row["sheet_name"] for row in rows_payload})
    else:
        sheets = parse_tabular(content, url, content_type)
        sheet_names = [str(name_) for name_ in sheets]
        for sheet_name, dataframe in sheets.items():
            for index, row in dataframe.iterrows():
                values = {f"c{i + 1}": clean_cell(value) for i, value in enumerate(row.tolist())}
                values = {key: value for key, value in values.items() if value is not None}
                if values:
                    rows_payload.append(
                        {
                            "source": source,
                            "dataset_key": dataset_key,
                            "sheet_name": str(sheet_name),
                            "row_number": int(index) + 1,
                            "row_data": values,
                            "synced_at": now(),
                        }
                    )

    if not rows_payload:
        raise RuntimeError("Dataset parsed to zero rows")

    sb.table("official_market_datasets").upsert(
        {
            "source": source,
            "dataset_key": dataset_key,
            "dataset_name": name,
            "source_page": source_page,
            "file_url": url,
            "file_sha256": sha,
            "last_synced_at": now(),
            "raw_meta": {
                "sheets": sheet_names,
                "bytes": len(content),
                "parser": parser,
                "content_type": content_type,
            },
        },
        on_conflict="source,dataset_key",
    ).execute()
    sb.table("official_market_rows").delete().eq("source", source).eq("dataset_key", dataset_key).execute()
    for start in range(0, len(rows_payload), 100):
        sb.table("official_market_rows").insert(rows_payload[start : start + 100]).execute()
    return 1, len(rows_payload)


def sync_official_source(sb, source: str, datasets: list[tuple[str, str, str]]) -> None:
    started = now()
    dataset_count = 0
    row_count = 0
    sb.table("official_market_sync_state").upsert(
        {"source": source, "status": "running", "last_started_at": started, "updated_at": started},
        on_conflict="source",
    ).execute()
    error_messages: list[str] = []
    headers = {"User-Agent": UA, "Accept-Language": "en-GB,en;q=0.9"}
    with httpx.Client(timeout=75, follow_redirects=True, headers=headers) as client:
        for prefix, name, page_url in datasets:
            try:
                downloads = discover_downloads(client, page_url, follow_detail_pages=source == "dls")
                log(f"OFFICIAL {source}/{prefix}: candidates={len(downloads)}")
                for url in downloads:
                    response = sync_get(client, url, retries=4)
                    if response is None or len(response.content) < 100:
                        continue
                    content_type = response.headers.get("content-type", "")
                    if "text/html" in content_type.lower() and not str(response.url).lower().split("?", 1)[0].endswith(EXTS):
                        continue
                    try:
                        datasets_written, rows_written = write_official_dataset(
                            sb,
                            source=source,
                            prefix=prefix,
                            name=name,
                            source_page=page_url,
                            url=str(response.url),
                            content=response.content,
                            content_type=content_type,
                        )
                        dataset_count += datasets_written
                        row_count += rows_written
                        log(f"  imported {rows_written:,} rows: {response.url}")
                    except Exception as exc:
                        message = f"{url}: {type(exc).__name__}: {exc}"
                        error_messages.append(message)
                        log(f"  official candidate skipped: {message}")
                    time.sleep(1)
            except Exception as exc:
                message = f"{prefix}: {type(exc).__name__}: {exc}"
                error_messages.append(message)
                log(f"OFFICIAL {source}/{prefix} failed; continuing: {message}")

    completed = now()
    sb.table("official_market_sync_state").upsert(
        {
            "source": source,
            "status": "done" if dataset_count else "partial",
            "datasets_written": dataset_count,
            "rows_written": row_count,
            "last_error": " | ".join(error_messages)[-2000:] if error_messages else None,
            "last_started_at": started,
            "last_completed_at": completed,
            "updated_at": completed,
        },
        on_conflict="source",
    ).execute()
    log(f"OFFICIAL {source}: datasets={dataset_count} rows={row_count:,} errors={len(error_messages)}")


def official_pipeline(sb) -> None:
    sync_official_source(
        sb,
        "cbc",
        [("rppi", "CBC Residential Property Price Indices", CBC_RPPI_PAGE)],
    )
    sync_official_source(
        sb,
        "dls",
        [(prefix, name, page) for prefix, (name, page) in DLS_STATS.items()],
    )
    sync_official_source(
        sb,
        "cystat",
        [("construction_cost_m2", "CYSTAT Cost per Square Metre of Completed Private Buildings", CYSTAT_COST_PAGE)],
    )


# -----------------------------------------------------------------------------
# Main persistent orchestrator
# -----------------------------------------------------------------------------


async def run_cycle(sb, cycle_number: int) -> None:
    log(f"\n=== PLANA DATA V5 cycle {cycle_number} started {now()} ===")

    for status in ("sale", "rent"):
        try:
            await market_pipeline(sb, status)
        except Exception as exc:
            log(f"MARKET {status} pipeline failed; continuing: {type(exc).__name__}: {exc}")
            state_put(sb, f"market_{status}", status="error", last_error=f"{type(exc).__name__}: {exc}")

    if cycle_number == 1 or cycle_number % DLS_EVERY_CYCLES == 0:
        try:
            await dls_pipeline(sb)
        except Exception as exc:
            log(f"DLS pipeline failed; continuing: {type(exc).__name__}: {exc}")

    if cycle_number == 1 or cycle_number % OFFICIAL_EVERY_CYCLES == 0:
        try:
            await asyncio.to_thread(official_pipeline, sb)
        except Exception as exc:
            log(f"OFFICIAL pipeline failed; continuing: {type(exc).__name__}: {exc}")

    log(f"=== PLANA DATA V5 cycle {cycle_number} complete ===")


async def main() -> None:
    load_dotenv()
    missing = [name for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY") if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

    log("PLANA DATA WORKER V5 starting")
    log(
        "market_pages_per_cycle="
        f"{MARKET_PAGES_PER_CYCLE} market_max_page={MARKET_MAX_PAGE} "
        f"dls_layers={ACTIVE_DLS_LAYERS} cycle_seconds={DATA_CYCLE_SECONDS}"
    )

    cycle_number = 1
    while True:
        try:
            await run_cycle(sb, cycle_number)
        except Exception as exc:
            log(f"UNEXPECTED CYCLE ERROR: {type(exc).__name__}: {exc}")
        cycle_number += 1
        log(f"Sleeping {DATA_CYCLE_SECONDS}s before next PLANA DATA cycle.")
        await asyncio.sleep(DATA_CYCLE_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
