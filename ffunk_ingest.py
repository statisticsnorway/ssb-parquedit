"""Bulk ingest processor for Altinn XML forms into parquedit/DuckLake.

DESIGN RATIONALE
----------------
The original per-form approach (DefaultFormProcessor.process_new_forms) calls
insert once per form per table, producing O(forms × tables) = ~5 000–7 000
individual insert calls for ~1 000 forms.  Each call lands as its own tiny
parquet file in DuckLake, which compounds into progressive query slowdown.

This module replaces that with a batched strategy:

  1. Discover all (xml, json-metadata) file pairs on disk.
  2. Parse and extract EVERY form into memory in one pass.
  3. Deduplicate in memory (within-batch) and against the DB in ONE bulk query.
  4. Build one pandas DataFrame per table.
  5. Call ParquEdit.insert_data() ONCE per table → 5 inserts total.

Expected wins
-------------
* ~1 000 forms × 5 tables   →  5 insert calls          (was ~5 000–7 000)
* 1 DB round-trip for dedup  →  1 SELECT on skjemamottak (was ~1 000 SELECTs)
* Each table becomes one (or very few) parquet files instead of thousands.

Limitations / future work
--------------------------
* checkbox_mapping is not supported (no standard parquedit table for it yet).
* enheter / enhetsinfo are only deduplicated within the batch, not against the
  DB.  Run compact_files.py afterwards to merge any accumulated unit duplicates.
* For very large corpora (> 100 k forms) the in-memory refnr set may need to be
  replaced by a DuckDB JOIN inside the parquedit connection.
"""

from __future__ import annotations

import glob
import logging
import requests
from pathlib import Path

import pandas as pd
import xmltodict

from ssb_altinn_form_tools.default_form_extractor import DefaultFormExtractor
from ssb_altinn_form_tools.models import ExtractedForm, FormJsonData
from ssb_parquedit import ParquEdit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table schemas
# The _id column is added automatically by parquedit; do NOT include it here.
# ---------------------------------------------------------------------------

TABLE_SCHEMAS: dict[str, dict] = {
    "skjemamottak": {
        "properties": {
            "aar":          {"type": "integer"},
            "skjema":       {"type": "string"},
            "ident":        {"type": "string"},
            "refnr":        {"type": "string"},
            "kommentar":    {"type": "string"},
            "dato_mottatt": {"type": "string", "format": "date-time"},
            "editert":      {"type": "string"},
            "aktiv":        {"type": "string"},
        },
        "required": ["aar", "skjema", "ident", "refnr", "dato_mottatt", "editert", "aktiv"],
    },
    "enheter": {
        "properties": {
            "aar":    {"type": "integer"},
            "ident":  {"type": "string"},
            "skjema": {"type": "string"},
        },
        "required": ["aar", "ident", "skjema"],
    },
    "kontaktinfo": {
        "properties": {
            "aar":                   {"type": "integer"},
            "skjema":                {"type": "string"},
            "ident":                 {"type": "string"},
            "refnr":                 {"type": "string"},
            "kontaktperson":         {"type": "string"},
            "epost":                 {"type": "string"},
            "telefon":               {"type": "string"},
            "bekreftet_kontaktinfo": {"type": "string"},
            "kommentar_kontaktinfo": {"type": "string"},
            "kommentar_krevende":    {"type": "string"},
        },
        "required": ["aar", "skjema", "ident", "refnr"],
    },
    "enhetsinfo": {
        "properties": {
            "aar":      {"type": "integer"},
            "ident":    {"type": "string"},
            "variabel": {"type": "string"},
            "verdi":    {"type": "string"},
        },
        "required": ["aar", "ident", "variabel"],
    },
    "skjemadata": {
        "properties": {
            "aar":      {"type": "integer"},
            "skjema":   {"type": "string"},
            "ident":    {"type": "string"},
            "refnr":    {"type": "string"},
            "feltsti":  {"type": "string"},
            "feltnavn": {"type": "string"},
            "verdi":    {"type": "string"},
            "alias":    {"type": "string"},
            "dybde":    {"type": "integer"},
            "indeks":   {"type": "integer"},
        },
        "required": ["aar", "skjema", "ident", "refnr", "feltsti", "feltnavn"],
    },
}

# Insertion order: reception first so refnr exists before dependant tables.
INSERT_ORDER = ["skjemamottak", "enheter", "kontaktinfo", "enhetsinfo", "skjemadata"]


# ---------------------------------------------------------------------------
# Altinn schema helpers
# ---------------------------------------------------------------------------

def _extract_arr_fields(schema: dict, parent: str | None = None) -> list[str]:
    """Recursively collect field names whose type is "array" in an Altinn JSON schema."""
    items: list[str] = []
    for key, value in schema.items():
        if isinstance(value, dict):
            items.extend(_extract_arr_fields(value, key))
        elif value == "array" and parent:
            items.append(parent)
    return items


def fetch_array_fields(form_name: str, ra_version: int = 1) -> list[str] | None:
    """Return array-type field names from the Altinn JSON schema endpoint.

    These are forwarded to ``xmltodict.parse(force_list=...)`` so that repeated
    XML elements are always returned as lists, even when only one child exists.

    Returns ``None`` on network/parse failures; callers should proceed without.
    """
    # Mirror the URL construction used by DefaultFormProcessor._get_json_schema
    ra_nummer   = f"{form_name[:2]}-{form_name[2:]}A3"  # "RA-0187A3"
    ra_base     = ra_nummer.split("A3")[0]               # "RA-0187"
    ra_id       = ra_base.replace("-", "").lower()       # "ra0187"
    version_str = f"{ra_version:02d}"                    # "01"
    url = (
        f"https://ssb.apps.altinn.no/ssb/{ra_id}-{version_str}"
        f"/api/jsonschema/A3_{ra_base}_M"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return _extract_arr_fields(resp.json())
    except Exception as exc:
        logger.warning("Could not fetch array fields from Altinn (%s): %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_form_pairs(
    base_path: str,
    glob_pattern: str | None = None,
) -> list[tuple[Path, Path]]:
    """Return ``(xml_path, json_metadata_path)`` pairs found under *base_path*.

    The metadata JSON sits alongside each XML file with the same stem but
    suffixes ``meta.json`` instead of ``form.xml`` (standard SUV layout).
    Pairs where the JSON companion is missing are skipped with a warning.
    """
    pattern = glob_pattern or f"{base_path}/**/**/**/**/*.xml"
    pairs: list[tuple[Path, Path]] = []
    for xml_file in glob.glob(pattern, recursive=True):
        xml_path  = Path(xml_file)
        json_name = xml_path.name.replace("xml", "json").replace("form", "meta")
        json_path = xml_path.with_name(json_name)
        if json_path.exists():
            pairs.append((xml_path, json_path))
        else:
            logger.warning("Missing JSON metadata for %s – skipping", xml_path.name)
    return pairs


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_one(
    xml_path: Path,
    json_path: Path,
    extractor: DefaultFormExtractor,
    form_data_key: str,
    array_fields: list[str] | None,
    alias_mapping: dict[str, str] | None,
) -> ExtractedForm | None:
    """Parse a single (xml, json) pair and return an ExtractedForm, or None on error."""
    try:
        json_data  = FormJsonData.model_validate_json(json_path.read_text())
        dictionary = xmltodict.parse(
            xml_path.read_text(), force_list=array_fields
        )[form_data_key]
        form = extractor.extract_form(dictionary, json_data)
        if alias_mapping:
            for fd in form.form_data:
                if fd.feltnavn in alias_mapping:
                    fd.alias = alias_mapping[fd.feltnavn]
        return form
    except Exception as exc:
        logger.error("Failed to extract %s: %s", xml_path.name, exc)
        return None


def extract_all_forms(
    pairs: list[tuple[Path, Path]],
    extractor: DefaultFormExtractor,
    form_data_key: str,
    array_fields: list[str] | None,
    alias_mapping: dict[str, str] | None = None,
) -> list[ExtractedForm]:
    """Extract every file pair, skipping failures, deduplicating on ``refnr``."""
    results: list[ExtractedForm] = []
    seen_refnr: set[str] = set()
    for xml_path, json_path in pairs:
        form = _extract_one(
            xml_path, json_path, extractor, form_data_key, array_fields, alias_mapping
        )
        if form is None:
            continue
        refnr = form.reception.refnr
        if refnr in seen_refnr:
            logger.warning(
                "Duplicate refnr %s in file set – keeping first occurrence", refnr
            )
        else:
            seen_refnr.add(refnr)
            results.append(form)
    return results


# ---------------------------------------------------------------------------
# Convert extracted forms to DataFrames
# ---------------------------------------------------------------------------

def forms_to_dataframes(forms: list[ExtractedForm]) -> dict[str, pd.DataFrame]:
    """Turn a list of ExtractedForm objects into one DataFrame per table.

    Boolean fields (bekreftet_kontaktinfo, aktiv) are cast to strings to match
    the VARCHAR schema used by parquedit.  Optional integer columns (dybde,
    indeks) use pandas nullable Int64 to preserve None values through the Arrow
    conversion in DMLOperations.
    """
    rows: dict[str, list[dict]] = {t: [] for t in TABLE_SCHEMAS}

    for form in forms:
        r = form.reception

        # skjemamottak – one row per form
        receipt = r.model_dump()
        receipt["dato_mottatt"] = (
            receipt["dato_mottatt"].isoformat()
            if receipt.get("dato_mottatt") else None
        )
        receipt["aktiv"] = str(receipt["aktiv"])
        rows["skjemamottak"].append(receipt)

        # enheter – one row per form; batch-deduplicated later
        rows["enheter"].append(form.unit.model_dump())

        # kontaktinfo – one row per form
        ci = form.contact_info.model_dump()
        ci["bekreftet_kontaktinfo"] = str(ci["bekreftet_kontaktinfo"])
        rows["kontaktinfo"].append(ci)

        # enhetsinfo – one row per InternInfo "enhets*" key
        for ui in form.unit_info:
            rows["enhetsinfo"].append(ui.model_dump())

        # skjemadata – many rows per form (the bulk of the data)
        for fd in form.form_data:
            rows["skjemadata"].append(fd.model_dump())

    dfs: dict[str, pd.DataFrame] = {}
    for table, data in rows.items():
        df = pd.DataFrame(data) if data else pd.DataFrame()
        # Use nullable integer dtype so None values survive Arrow serialisation
        for col in ("dybde", "indeks"):
            if col in df.columns:
                df[col] = df[col].astype(pd.Int64Dtype())
        dfs[table] = df
    return dfs


# ---------------------------------------------------------------------------
# Database deduplication
# ---------------------------------------------------------------------------

def fetch_existing_refnr(conn: ParquEdit) -> set[str]:
    """Return all refnr already present in skjemamottak (single DB read)."""
    if not conn.exists("skjemamottak"):
        return set()
    try:
        df = conn.view("skjemamottak", columns=["refnr"])
        if df is None or df.empty:
            return set()
        return set(df["refnr"].dropna().tolist())
    except Exception as exc:
        logger.warning("Could not query existing refnr: %s", exc)
        return set()


def filter_new_dataframes(
    dataframes: dict[str, pd.DataFrame],
    existing_refnr: set[str],
) -> dict[str, pd.DataFrame]:
    """Remove rows that would be duplicates in the database.

    * Tables with a ``refnr`` column (skjemamottak, kontaktinfo, skjemadata):
      rows whose refnr is already in *existing_refnr* are dropped.
    * enheter: deduplicated within the batch by (aar, ident, skjema) so the
      same unit is not inserted multiple times for different forms in one run.
    * enhetsinfo: deduplicated within the batch by (aar, ident, variabel);
      only rows belonging to units with at least one new form are kept.
    """
    result: dict[str, pd.DataFrame] = {}

    # Filter refnr-bearing tables
    for table in ("skjemamottak", "kontaktinfo", "skjemadata"):
        df = dataframes.get(table, pd.DataFrame())
        if not df.empty and existing_refnr and "refnr" in df.columns:
            df = df[~df["refnr"].isin(existing_refnr)].reset_index(drop=True)
        result[table] = df

    # enheter: batch dedup
    enheter_df = dataframes.get("enheter", pd.DataFrame()).copy()
    if not enheter_df.empty:
        enheter_df = enheter_df.drop_duplicates(
            subset=["aar", "ident", "skjema"]
        ).reset_index(drop=True)
    result["enheter"] = enheter_df

    # enhetsinfo: keep only rows belonging to units that have new forms,
    # then dedup within the batch
    enhetsinfo_df = dataframes.get("enhetsinfo", pd.DataFrame()).copy()
    if not enhetsinfo_df.empty:
        new_skjemamottak = result.get("skjemamottak", pd.DataFrame())
        if not new_skjemamottak.empty:
            new_ident_aar: set[tuple] = set(
                zip(new_skjemamottak["ident"], new_skjemamottak["aar"])
            )
            mask = enhetsinfo_df.apply(
                lambda row: (row["ident"], row["aar"]) in new_ident_aar, axis=1
            )
            enhetsinfo_df = enhetsinfo_df[mask]
        enhetsinfo_df = enhetsinfo_df.drop_duplicates(
            subset=["aar", "ident", "variabel"]
        ).reset_index(drop=True)
    result["enhetsinfo"] = enhetsinfo_df

    return result


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def ensure_tables(conn: ParquEdit, product_name: str) -> None:
    """Create any tables that do not yet exist (idempotent)."""
    for table, schema in TABLE_SCHEMAS.items():
        if not conn.exists(table):
            logger.info("Creating table: %s", table)
            conn.create_table(table, schema, product_name=product_name)
        else:
            logger.debug("Table already exists: %s", table)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def bulk_ingest(
    conn: ParquEdit,
    form_name: str,
    form_base_path: str,
    product_name: str,
    alias_mapping: dict[str, str] | None = None,
    ra_version: int = 1,
    glob_pattern: str | None = None,
) -> dict[str, int]:
    """Bulk-ingest all new Altinn XML forms into parquedit/DuckLake tables.

    This replaces DefaultFormProcessor.process_new_forms() with a strategy that
    accumulates all data in memory and issues a single insert per table instead of
    one insert per form per table.

    Parameters
    ----------
    conn:
        A configured ParquEdit instance.
    form_name:
        Altinn form code, e.g. ``"RA0187"``.
    form_base_path:
        Root directory where XML form files (and their JSON companions) reside.
    product_name:
        Label used when creating new tables (required by parquedit).
        Must be identical across runs so table creation stays idempotent.
    alias_mapping:
        Optional ``{feltnavn: alias}`` mapping applied to form data rows.
        Example: ``{"omsVirksomhetPerioden": "omsetning"}``.
    ra_version:
        Altinn schema version integer (default 1).  Used to build the
        JSON-schema URL for ``force_list`` discovery.
    glob_pattern:
        Override the default glob used to discover XML files.  The default
        follows the standard SUV directory layout:
        ``<form_base_path>/**/**/**/**/*.xml``.

    Returns
    -------
    dict mapping table name → number of rows inserted in this run.
    Zero-counts are included so callers can confirm all tables were visited.
    """
    form_data_key = f"A3_{form_name}_M"
    logger.info("bulk_ingest: starting for %s", form_data_key)

    # 1. Fetch array-field metadata so xmltodict always returns lists
    array_fields = fetch_array_fields(form_name, ra_version)

    # 2. Discover (xml, json) file pairs
    pairs = find_form_pairs(form_base_path, glob_pattern)
    if not pairs:
        logger.warning("No XML form files found under %s", form_base_path)
        return {}
    logger.info("Found %d file pairs", len(pairs))

    # 3 & 4. Parse every file; batch-deduplicate on refnr in memory
    extractor = DefaultFormExtractor()
    forms = extract_all_forms(
        pairs, extractor, form_data_key, array_fields, alias_mapping
    )
    logger.info(
        "Extracted %d unique forms from %d file pairs (%d failed/duplicate)",
        len(forms), len(pairs), len(pairs) - len(forms),
    )
    if not forms:
        return {}

    # 5. Build one DataFrame per table
    dataframes = forms_to_dataframes(forms)

    # 6. One DB read to discover already-ingested refnr
    existing_refnr = fetch_existing_refnr(conn)
    logger.info("DB already contains %d known refnr", len(existing_refnr))

    # 7. Filter out existing data
    new_dfs = filter_new_dataframes(dataframes, existing_refnr)

    new_form_count = len(new_dfs.get("skjemamottak", pd.DataFrame()))
    skipped = len(forms) - new_form_count
    logger.info(
        "%d new forms to insert, %d already in DB (skipped)", new_form_count, skipped
    )
    if new_form_count == 0:
        logger.info("Nothing to insert – all forms already present.")
        return {t: 0 for t in TABLE_SCHEMAS}

    # 8. Create tables if needed
    ensure_tables(conn, product_name)

    # 9. One bulk insert per table
    rows_inserted: dict[str, int] = {}
    for table in INSERT_ORDER:
        df = new_dfs.get(table, pd.DataFrame())
        if df.empty:
            rows_inserted[table] = 0
            logger.info("  %-20s  0 rows", table)
            continue
        logger.info("  %-20s  inserting %d rows …", table, len(df))
        conn.insert_data(table, df)
        rows_inserted[table] = len(df)
        logger.info("  %-20s  done", table)

    total = sum(rows_inserted.values())
    logger.info(
        "bulk_ingest complete: %d total rows across %d tables",
        total, len(INSERT_ORDER),
    )
    return rows_inserted


# ---------------------------------------------------------------------------
# Direct execution – mirrors ingest.py usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    conn = ParquEdit()

    result = bulk_ingest(
        conn=conn,
        form_name="RA0187",
        form_base_path="/home/onyxia/work/RA0187",
        product_name="vhi",
        alias_mapping={"omsVirksomhetPerioden": "omsetning"},
    )

    print("\nRows inserted per table:")
    for table, count in result.items():
        print(f"  {table:22s}  {count:>8d} rows")

    print("\nenheter preview (first 10 rows):")
    print(conn.view("enheter", limit=10))
