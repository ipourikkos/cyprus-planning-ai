"""PLANA.CY Direct ArcGIS Parcel Indexer V2.

Render command:
    python indexer.py

Required environment variables:
    SUPABASE_URL
    SUPABASE_SECRET_KEY

Optional environment variables:
    PLANA_INDEX_PAGE_SIZE=1000
    PLANA_INDEX_STALE_DAYS=30
    PLANA_INDEX_SLEEP_SECONDS=1.0
    PLANA_INDEX_CYCLE_SECONDS=86400
    PLANA_INDEX_MAX_PAGES=0

Architecture:
- Pages directly through the official DLS ArcGIS Parcels layer.
- Loads Planning Zones and administrative code/name lookups from the official
  DLS ArcGIS map service.
- Spatially joins parcel polygons to planning-zone polygons locally.
- Reuses planning coefficients already learned in plana_parcels by zone code
  (median of existing non-null indexed values).
- Never calls GeneralParcelIdentify in the mass-indexing path.
- Does not import app.py; worker deploys are therefore decoupled from the web app.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv
from pyproj import Geod
from shapely.geometry import shape
from shapely.strtree import STRtree
from supabase import create_client

DLS_MAPSERVER = "https://eservices.dls.moi.gov.cy/arcgis/rest/services/National/CadastralMap_EN/MapServer"
PARCEL_QUERY = f"{DLS_MAPSERVER}/0/query"
ZONE_QUERY = f"{DLS_MAPSERVER}/12/query"
DISTRICT_QUERY = f"{DLS_MAPSERVER}/15/query"
MUNICIPALITY_QUERY = f"{DLS_MAPSERVER}/16/query"
QUARTER_QUERY = f"{DLS_MAPSERVER}/17/query"

GEOD = Geod(ellps="WGS84")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def force_2d_geometry(geom: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(geom, dict):
        return None

    def xy(value: Any) -> Any:
        if isinstance(value, list):
            if value and all(isinstance(item, (int, float)) for item in value):
                return value[:2]
            return [xy(item) for item in value]
        return value

    return {**geom, "coordinates": xy(geom.get("coordinates"))}


def geometry_metrics(geom_json: dict[str, Any]) -> dict[str, float | None]:
    geom = shape(force_2d_geometry(geom_json) or geom_json)
    centroid = geom.representative_point()

    try:
        signed_area, perimeter = GEOD.geometry_area_perimeter(geom)
        area_m2 = abs(float(signed_area))
        perimeter_m = abs(float(perimeter))
    except Exception:
        area_m2 = 0.0
        perimeter_m = 0.0

    edge_lengths: list[float] = []
    edge_bearings: list[float] = []
    polygons = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for polygon in polygons:
        coords = list(polygon.exterior.coords)
        for first, second in zip(coords, coords[1:]):
            lon1, lat1 = first[:2]
            lon2, lat2 = second[:2]
            azimuth, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
            if distance > 0:
                edge_lengths.append(float(distance))
                edge_bearings.append(float(azimuth % 180.0))

    longest = max(edge_lengths) if edge_lengths else None
    shortest = min(edge_lengths) if edge_lengths else None
    orientation = None
    if longest is not None:
        index = edge_lengths.index(longest)
        orientation = edge_bearings[index]

    compactness = None
    if area_m2 > 0 and perimeter_m > 0:
        compactness = clamp(4.0 * math.pi * area_m2 / (perimeter_m * perimeter_m), 0.0, 1.0) * 100.0

    return {
        "centroid_lat": round(float(centroid.y), 8),
        "centroid_lon": round(float(centroid.x), 8),
        "area_m2": round(area_m2, 2) if area_m2 else None,
        "perimeter_m": round(perimeter_m, 2) if perimeter_m else None,
        "longest_edge_m": round(longest, 2) if longest else None,
        "shortest_edge_m": round(shortest, 2) if shortest else None,
        "orientation_deg": round(orientation, 1) if orientation is not None else None,
        "compactness": round(compactness, 1) if compactness is not None else None,
    }


async def arcgis_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = 5,
    timeout: float = 30.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict) and not payload.get("error"):
                    return payload
                raise RuntimeError(f"ArcGIS payload error: {payload.get('error') if isinstance(payload, dict) else payload!r}")
            raise RuntimeError(f"HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = min(2 ** attempt, 30)
            print(f"ArcGIS retry {attempt}/{attempts} url={url.rsplit('/', 2)[-2]} delay={delay}s error={exc!r}", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError(f"ArcGIS request failed after {attempts} attempts: {last_error!r}")


async def fetch_all_features(
    client: httpx.AsyncClient,
    url: str,
    *,
    out_fields: str,
    return_geometry: bool,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "f": "geojson",
            "where": "1=1",
            "outFields": out_fields,
            "returnGeometry": str(return_geometry).lower(),
            "outSR": "4326",
            "orderByFields": "OBJECTID ASC",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        payload = await arcgis_json(client, url, params)
        page = payload.get("features") or []
        features.extend(page)
        print(f"loaded reference layer={url.rsplit('/', 2)[-2]} offset={offset} rows={len(page)} total={len(features)}", flush=True)
        if not page or (len(page) < page_size and not payload.get("exceededTransferLimit")):
            break
        offset += len(page)
    return features


def property_dict(feature: dict[str, Any]) -> dict[str, Any]:
    return feature.get("properties") or feature.get("attributes") or {}


def lookup_from_features(
    features: Iterable[dict[str, Any]],
    code_fields: tuple[str, ...],
    name_field: str,
) -> dict[tuple[int, ...], str]:
    result: dict[tuple[int, ...], str] = {}
    for feature in features:
        props = property_dict(feature)
        try:
            key = tuple(int(props[field]) for field in code_fields)
        except Exception:
            continue
        name = clean_text(props.get(name_field))
        if name:
            result[key] = name
    return result


async def load_zone_coefficients(sb: Any) -> dict[str, dict[str, float | None]]:
    """Learn stable zone coefficients from already enriched rows in Supabase.

    Existing index rows created from detailed DLS responses may already contain
    density/coverage/floor/height values. We take the median per individual zone
    code and reuse those coefficients for direct ArcGIS indexing.
    """
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    offset = 0
    page_size = 1000
    while True:
        response = (
            sb.table("plana_parcels")
            .select("planning_zone,density_percent,coverage_percent,max_floors,max_height_m")
            .not_.is_("planning_zone", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            zone_text = clean_text(row.get("planning_zone"))
            if not zone_text:
                continue
            zones = [part.strip() for part in zone_text.split("/") if part.strip()]
            for zone in zones:
                for field in ("density_percent", "coverage_percent", "max_floors", "max_height_m"):
                    value = num(row.get(field))
                    if value is not None:
                        grouped[zone][field].append(value)
        if len(rows) < page_size:
            break
        offset += page_size

    result: dict[str, dict[str, float | None]] = {}
    for zone, fields in grouped.items():
        result[zone] = {
            field: round(float(statistics.median(values)), 2) if values else None
            for field, values in fields.items()
        }
    print(f"learned coefficients for {len(result)} planning zones from existing PLANA rows", flush=True)
    return result


class ZoneIndex:
    def __init__(self, features: list[dict[str, Any]]):
        self.features: list[dict[str, Any]] = []
        self.geometries = []
        for feature in features:
            geom_json = force_2d_geometry(feature.get("geometry"))
            if not geom_json:
                continue
            try:
                geom = shape(geom_json)
            except Exception:
                continue
            if geom.is_empty:
                continue
            self.features.append(feature)
            self.geometries.append(geom)
        self.tree = STRtree(self.geometries)

    def zones_for(self, parcel_geom_json: dict[str, Any], parcel_area_m2: float | None) -> list[dict[str, Any]]:
        parcel_geom = shape(force_2d_geometry(parcel_geom_json) or parcel_geom_json)
        candidates = self.tree.query(parcel_geom)
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            if hasattr(candidate, "item"):
                index = int(candidate.item())
                zone_geom = self.geometries[index]
                feature = self.features[index]
            elif isinstance(candidate, int):
                index = candidate
                zone_geom = self.geometries[index]
                feature = self.features[index]
            else:
                zone_geom = candidate
                try:
                    index = self.geometries.index(candidate)
                except ValueError:
                    continue
                feature = self.features[index]
            if not zone_geom.intersects(parcel_geom):
                continue
            props = property_dict(feature)
            name = clean_text(props.get("PLNZNT_NAME"))
            if not name:
                continue
            overlap_percent = None
            try:
                intersection = zone_geom.intersection(parcel_geom)
                signed_area, _ = GEOD.geometry_area_perimeter(intersection)
                intersection_area = abs(float(signed_area))
                if parcel_area_m2 and parcel_area_m2 > 0:
                    overlap_percent = round(clamp(intersection_area / parcel_area_m2 * 100.0), 2)
            except Exception:
                pass
            rows.append({
                "zone": name,
                "description": clean_text(props.get("PLNZNT_DESC")),
                "overlap_percent": overlap_percent,
            })
        rows.sort(key=lambda item: item.get("overlap_percent") or 0.0, reverse=True)
        return rows


def coefficient_for_zones(
    zones: list[dict[str, Any]],
    zone_coefficients: dict[str, dict[str, float | None]],
) -> tuple[dict[str, float | None], str]:
    if not zones:
        return {"density_percent": None, "coverage_percent": None, "max_floors": None, "max_height_m": None}, "missing"

    weighted: dict[str, list[tuple[float, float]]] = defaultdict(list)
    matched = 0
    for zone in zones:
        coeff = zone_coefficients.get(zone["zone"])
        if not coeff:
            continue
        matched += 1
        weight = (num(zone.get("overlap_percent")) or 0.0) / 100.0
        if weight <= 0:
            weight = 1.0 / max(len(zones), 1)
        for field, value in coeff.items():
            if value is not None:
                weighted[field].append((float(value), weight))

    output: dict[str, float | None] = {}
    for field in ("density_percent", "coverage_percent", "max_floors", "max_height_m"):
        values = weighted.get(field) or []
        if not values:
            output[field] = None
            continue
        total_weight = sum(weight for _, weight in values)
        output[field] = round(sum(value * weight for value, weight in values) / total_weight, 2)
    status = "matched_all" if matched == len(zones) else ("matched_partial" if matched else "missing")
    return output, status


def status_and_units(area: float | None, density: float | None, floors: float | None) -> dict[str, Any]:
    capacity = area * density / 100.0 if area and density is not None else None
    apt_low = apt_high = house_low = house_high = None
    apartment_status = house_status = mixed_status = "screening_pending_zone_coefficients"
    best_use = "Parcel screening"
    if capacity and capacity > 0:
        apt_mid = capacity * 0.825 / 85.0
        apt_low = max(int(math.floor(apt_mid * 0.85)), 1)
        apt_high = max(int(math.ceil(apt_mid * 1.15)), apt_low)
        house_mid = capacity * 0.78 / 180.0
        house_low = max(int(math.floor(house_mid * 0.85)), 1)
        house_high = max(int(math.ceil(house_mid * 1.15)), house_low)
        house_status = "preliminary_capacity_fit"
        if (floors or 0) >= 2 and density >= 60:
            apartment_status = "preliminary_capacity_fit"
            best_use = "Apartments" if apt_high >= 4 else "Residential development"
        else:
            apartment_status = "lower_density_screen"
            best_use = "Houses"
        mixed_status = "conditional_use_screen"
    return {
        "floor_capacity_m2": round(capacity, 1) if capacity else None,
        "apartment_units_low": apt_low,
        "apartment_units_high": apt_high,
        "house_units_low": house_low,
        "house_units_high": house_high,
        "apartment_status": apartment_status,
        "house_status": house_status,
        "mixed_use_status": mixed_status,
        "best_use": best_use,
    }


def preliminary_scores(
    *,
    area: float | None,
    compactness: float | None,
    shape_ratio: float | None,
    density: float | None,
    coverage: float | None,
    floors: float | None,
    zone_status: str,
) -> dict[str, float | None]:
    planning = 35.0
    if zone_status == "matched_all":
        planning = 84.0
    elif zone_status == "matched_partial":
        planning = 70.0
    elif density is not None:
        planning = 76.0
    elif zone_status != "missing":
        planning = 54.0

    density_signal = clamp((density or 0.0) / 1.4, 0.0, 100.0)
    floor_signal = clamp((floors or 0.0) * 18.0, 0.0, 100.0)
    scale_signal = 50.0
    if area:
        # Broad sweet spot for development screening, without forcing a single scale.
        log_distance = abs(math.log(max(area, 1.0)) - math.log(1800.0))
        scale_signal = clamp(100.0 - log_distance * 28.0, 25.0, 100.0)
    development = round(0.45 * density_signal + 0.25 * floor_signal + 0.30 * scale_signal, 1)

    compact_signal = compactness if compactness is not None else 50.0
    ratio_signal = clamp((shape_ratio or 0.35) * 130.0, 20.0, 100.0)
    site = round(0.55 * compact_signal + 0.45 * ratio_signal, 1)

    opportunity = round(0.36 * planning + 0.39 * development + 0.25 * site, 1)
    confidence = 72.0 if zone_status == "matched_all" else (58.0 if zone_status == "matched_partial" else (44.0 if zone_status != "missing" else 30.0))
    risk_adjusted = round(opportunity * (0.80 + 0.20 * confidence / 100.0), 1)
    return {
        "planning_score": round(planning, 1),
        "development_score": development,
        "site_score": site,
        "market_score": None,
        "financial_score": None,
        "opportunity_score": opportunity,
        "data_confidence": confidence,
        "risk_adjusted_score": risk_adjusted,
    }


def parcel_row(
    feature: dict[str, Any],
    *,
    zone_index: ZoneIndex,
    zone_coefficients: dict[str, dict[str, float | None]],
    district_lookup: dict[tuple[int, ...], str],
    municipality_lookup: dict[tuple[int, ...], str],
    quarter_lookup: dict[tuple[int, ...], str],
) -> dict[str, Any] | None:
    props = property_dict(feature)
    geom_json = force_2d_geometry(feature.get("geometry"))
    if not geom_json:
        return None
    try:
        parcel_id = int(float(props.get("SBPI_ID_NO")))
    except Exception:
        return None

    metrics = geometry_metrics(geom_json)
    area = num(props.get("SHAPE.STArea()")) or num(props.get("Parcel Extend")) or num(metrics.get("area_m2"))
    zones = zone_index.zones_for(geom_json, area)
    coeff, coeff_status = coefficient_for_zones(zones, zone_coefficients)

    density = num(coeff.get("density_percent"))
    coverage = num(coeff.get("coverage_percent"))
    floors = num(coeff.get("max_floors"))
    max_height = num(coeff.get("max_height_m"))
    longest = num(metrics.get("longest_edge_m"))
    shortest = num(metrics.get("shortest_edge_m"))
    shape_ratio = round(shortest / longest, 4) if longest and shortest else None
    compactness = num(metrics.get("compactness"))

    capacity = status_and_units(area, density, floors)
    score = preliminary_scores(
        area=area,
        compactness=compactness,
        shape_ratio=shape_ratio,
        density=density,
        coverage=coverage,
        floors=floors,
        zone_status=coeff_status if zones else "missing",
    )

    try:
        dist_code = int(props.get("DIST_CODE"))
    except Exception:
        dist_code = None
    try:
        vil_code = int(props.get("VIL_CODE"))
    except Exception:
        vil_code = None
    try:
        qrtr_code = int(props.get("QRTR_CODE"))
    except Exception:
        qrtr_code = None

    district = district_lookup.get((dist_code,)) if dist_code is not None else None
    municipality = municipality_lookup.get((dist_code, vil_code)) if dist_code is not None and vil_code is not None else None
    if municipality is None and vil_code is not None:
        municipality = municipality_lookup.get((vil_code,))
    quarter = quarter_lookup.get((dist_code, vil_code, qrtr_code)) if None not in (dist_code, vil_code, qrtr_code) else None
    if quarter is None and qrtr_code is not None:
        quarter = quarter_lookup.get((qrtr_code,))

    zone_text = " / ".join(zone["zone"] for zone in zones) or None
    overlap_total = sum(num(zone.get("overlap_percent")) or 0.0 for zone in zones) if zones else None
    now = datetime.now(timezone.utc).isoformat()

    return {
        "parcel_id": parcel_id,
        "parcel_number": props.get("PARCEL_NBR"),
        "district": district,
        "municipality": municipality,
        "quarter": quarter,
        "planning_zone": zone_text,
        "centroid_lat": metrics.get("centroid_lat"),
        "centroid_lon": metrics.get("centroid_lon"),
        "geom": geom_json,
        "parcel_area_m2": round(area, 1) if area else None,
        "perimeter_m": metrics.get("perimeter_m"),
        "longest_edge_m": metrics.get("longest_edge_m"),
        "shortest_edge_m": metrics.get("shortest_edge_m"),
        "shape_ratio": shape_ratio,
        "compactness": compactness,
        "orientation_deg": metrics.get("orientation_deg"),
        "density_percent": density,
        "coverage_percent": coverage,
        "max_floors": floors,
        "max_height_m": max_height,
        "floor_capacity_m2": capacity["floor_capacity_m2"],
        "coverage_capacity_m2": round(area * coverage / 100.0, 1) if area and coverage is not None else None,
        "existing_enclosed_m2": None,
        # Do not falsely label every directly indexed parcel as 100% underused.
        "development_gap_m2": None,
        "development_gap_percent": None,
        **score,
        "apartment_units_low": capacity["apartment_units_low"],
        "apartment_units_high": capacity["apartment_units_high"],
        "house_units_low": capacity["house_units_low"],
        "house_units_high": capacity["house_units_high"],
        "apartment_status": capacity["apartment_status"],
        "house_status": capacity["house_status"],
        "mixed_use_status": capacity["mixed_use_status"],
        "best_use": capacity["best_use"],
        "score_version": "direct-arcgis-index-v2",
        "raw_summary": {
            "index_source": "official_dls_arcgis_bulk",
            "coefficient_source": "existing_plana_zone_medians" if coeff_status != "missing" else "pending",
            "coefficient_match_status": coeff_status,
            "primary_zone_overlap_percent": zones[0].get("overlap_percent") if zones else None,
            "zone_overlap_total_percent": round(overlap_total, 2) if overlap_total is not None else None,
            "zones": zones,
            "sheet": clean_text(props.get("SHEET")),
            "plan": clean_text(props.get("PLAN_NBR")),
            "block": props.get("BLCK_CODE"),
            "district_code": dist_code,
            "municipality_code": vil_code,
            "quarter_code": qrtr_code,
        },
        "indexed_at": now,
        "dls_refreshed_at": now,
    }


async def upsert_rows(sb: Any, rows: list[dict[str, Any]]) -> tuple[int, int]:
    written = 0
    failed = 0
    for start in range(0, len(rows), 100):
        chunk = rows[start : start + 100]
        try:
            await asyncio.to_thread(
                lambda data=chunk: sb.table("plana_parcels").upsert(data, on_conflict="parcel_id").execute()
            )
            written += len(chunk)
        except Exception as exc:
            print(f"batch upsert failed rows={len(chunk)} error={exc!r}; retrying individually", flush=True)
            for item in chunk:
                try:
                    await asyncio.to_thread(
                        lambda data=item: sb.table("plana_parcels").upsert(data, on_conflict="parcel_id").execute()
                    )
                    written += 1
                except Exception as row_exc:
                    failed += 1
                    print(f"skip upsert parcel_id={item.get('parcel_id')} error={row_exc!r}", flush=True)
    return written, failed


async def run_cycle(sb: Any, client: httpx.AsyncClient) -> None:
    page_size = max(100, min(env_int("PLANA_INDEX_PAGE_SIZE", 1000), 1000))
    stale_days = max(1, env_int("PLANA_INDEX_STALE_DAYS", 30))
    sleep_seconds = max(0.0, env_float("PLANA_INDEX_SLEEP_SECONDS", 1.0))
    max_pages = max(0, env_int("PLANA_INDEX_MAX_PAGES", 0))
    cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)

    print("PLANA Direct ArcGIS Index V2 starting", flush=True)
    print(f"page_size={page_size} stale_days={stale_days} max_pages={max_pages or 'all'}", flush=True)
    print("loading DLS reference layers...", flush=True)

    district_features, municipality_features, quarter_features, zone_features = await asyncio.gather(
        fetch_all_features(client, DISTRICT_QUERY, out_fields="DIST_CODE,DIST_NM_E,OBJECTID", return_geometry=False),
        fetch_all_features(client, MUNICIPALITY_QUERY, out_fields="DIST_CODE,VIL_CODE,VIL_NM_E,OBJECTID", return_geometry=False),
        fetch_all_features(client, QUARTER_QUERY, out_fields="DIST_CODE,VIL_CODE,QRTR_CODE,QRTR_NM_E,OBJECTID", return_geometry=False),
        fetch_all_features(client, ZONE_QUERY, out_fields="PLNZNT_NAME,PLNZNT_DESC,OBJECTID", return_geometry=True),
    )

    district_lookup = lookup_from_features(district_features, ("DIST_CODE",), "DIST_NM_E")
    municipality_lookup = lookup_from_features(municipality_features, ("DIST_CODE", "VIL_CODE"), "VIL_NM_E")
    municipality_lookup.update(lookup_from_features(municipality_features, ("VIL_CODE",), "VIL_NM_E"))
    quarter_lookup = lookup_from_features(quarter_features, ("DIST_CODE", "VIL_CODE", "QRTR_CODE"), "QRTR_NM_E")
    quarter_lookup.update(lookup_from_features(quarter_features, ("QRTR_CODE",), "QRTR_NM_E"))
    zone_index = ZoneIndex(zone_features)
    zone_coefficients = await asyncio.to_thread(load_zone_coefficients_sync, sb)

    print(
        f"reference layers ready districts={len(district_lookup)} municipalities={len(municipality_lookup)} "
        f"quarters={len(quarter_lookup)} zones={len(zone_index.features)} coefficient_zones={len(zone_coefficients)}",
        flush=True,
    )

    offset = 0
    page_number = 0
    total_discovered = 0
    total_written = 0
    total_skipped_fresh = 0
    total_failed = 0

    while True:
        page_number += 1
        if max_pages and page_number > max_pages:
            break
        started = time.monotonic()
        params = {
            "f": "geojson",
            "where": "1=1",
            "outFields": "SBPI_ID_NO,DIST_CODE,VIL_CODE,QRTR_CODE,BLCK_CODE,PARCEL_NBR,SHEET,PLAN_NBR,SRC_SL_CODE,SHAPE.STArea(),OBJECTID",
            "returnGeometry": "true",
            "outSR": "4326",
            "orderByFields": "OBJECTID ASC",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        payload = await arcgis_json(client, PARCEL_QUERY, params, attempts=6, timeout=45.0)
        features = payload.get("features") or []
        if not features:
            print(f"parcel paging complete at offset={offset}", flush=True)
            break

        total_discovered += len(features)
        by_id: dict[int, dict[str, Any]] = {}
        for feature in features:
            props = property_dict(feature)
            try:
                parcel_id = int(float(props.get("SBPI_ID_NO")))
            except Exception:
                continue
            by_id[parcel_id] = feature

        ids = list(by_id)
        existing: dict[int, str | None] = {}
        for start in range(0, len(ids), 200):
            subset = ids[start : start + 200]
            response = await asyncio.to_thread(
                lambda values=subset: sb.table("plana_parcels")
                .select("parcel_id,indexed_at,score_version")
                .in_("parcel_id", values)
                .execute()
            )
            for row in response.data or []:
                try:
                    existing[int(row["parcel_id"])] = row.get("indexed_at")
                except Exception:
                    continue

        work: list[dict[str, Any]] = []
        skipped_fresh = 0
        for parcel_id, feature in by_id.items():
            stamp = existing.get(parcel_id)
            if stamp:
                try:
                    parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
                    if parsed > cutoff:
                        skipped_fresh += 1
                        continue
                except Exception:
                    pass
            work.append(feature)

        rows: list[dict[str, Any]] = []
        parse_failed = 0
        for feature in work:
            try:
                row = parcel_row(
                    feature,
                    zone_index=zone_index,
                    zone_coefficients=zone_coefficients,
                    district_lookup=district_lookup,
                    municipality_lookup=municipality_lookup,
                    quarter_lookup=quarter_lookup,
                )
                if row:
                    rows.append(row)
            except Exception as exc:
                parse_failed += 1
                props = property_dict(feature)
                print(f"skip parse parcel_id={props.get('SBPI_ID_NO')} error={exc!r}", flush=True)

        written, write_failed = await upsert_rows(sb, rows)
        total_written += written
        total_skipped_fresh += skipped_fresh
        total_failed += parse_failed + write_failed
        elapsed = time.monotonic() - started
        print(
            f"page {page_number} offset={offset} discovered={len(features)} work={len(work)} "
            f"fresh={skipped_fresh} written={written} failed={parse_failed + write_failed} "
            f"total_written={total_written} elapsed={elapsed:.1f}s",
            flush=True,
        )

        offset += len(features)
        if len(features) < page_size and not payload.get("exceededTransferLimit"):
            break
        if sleep_seconds:
            await asyncio.sleep(sleep_seconds)

    print(
        f"PLANA Direct ArcGIS Index V2 cycle complete discovered={total_discovered} "
        f"written={total_written} fresh_skipped={total_skipped_fresh} failed={total_failed}",
        flush=True,
    )


def load_zone_coefficients_sync(sb: Any) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    offset = 0
    page_size = 1000
    while True:
        response = (
            sb.table("plana_parcels")
            .select("planning_zone,density_percent,coverage_percent,max_floors,max_height_m")
            .not_.is_("planning_zone", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            zone_text = clean_text(row.get("planning_zone"))
            if not zone_text:
                continue
            zones = [part.strip() for part in zone_text.split("/") if part.strip()]
            for zone in zones:
                for field in ("density_percent", "coverage_percent", "max_floors", "max_height_m"):
                    value = num(row.get(field))
                    if value is not None:
                        grouped[zone][field].append(value)
        if len(rows) < page_size:
            break
        offset += page_size
    result: dict[str, dict[str, float | None]] = {}
    for zone, fields in grouped.items():
        result[zone] = {
            field: round(float(statistics.median(values)), 2) if values else None
            for field, values in fields.items()
        }
    print(f"learned coefficients for {len(result)} planning zones from existing PLANA rows", flush=True)
    return result


async def main() -> None:
    load_dotenv()
    sb = create_client(require_env("SUPABASE_URL"), require_env("SUPABASE_SECRET_KEY"))
    cycle_seconds = max(3600, env_int("PLANA_INDEX_CYCLE_SECONDS", 86400))
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(45.0, connect=10.0),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        follow_redirects=True,
        headers={
            "User-Agent": "PLANA.CY-Direct-ArcGIS-Indexer/2.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://eservices.dls.moi.gov.cy/",
        },
    ) as client:
        while True:
            try:
                await run_cycle(sb, client)
            except Exception as exc:
                print(f"index cycle failed: {exc!r}", flush=True)
                print("sleeping 900 seconds before retry", flush=True)
                await asyncio.sleep(900)
                continue
            print(f"index cycle complete; sleeping {cycle_seconds} seconds", flush=True)
            await asyncio.sleep(cycle_seconds)


if __name__ == "__main__":
    asyncio.run(main())
