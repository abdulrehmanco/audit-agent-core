"""
Audit Agent: Three-Way Matcher
Module 1 — The Ledger Parser
================================

This module ingests a client's General Ledger (GL) exported as an Excel
workbook (.xlsx) and converts it into a clean, predictable, machine-safe
list of dictionaries that downstream modules (e.g. the deterministic
three-way comparison engine) can consume without worrying about messy,
vendor-specific export formats.

Enterprise systems (SAP, Oracle, QuickBooks, NetSuite, ...) all export
the "same" concepts under wildly different column headers. The job of this
module is to:

    1. Read the workbook robustly.
    2. Normalize variable/messy column names into a fixed internal schema.
    3. Coerce values into safe, deterministic Python types
       (float for money, clean str for references).
    4. Preserve the *physical* Excel row number so an auditor can be told
       exactly which line in their original spreadsheet passed or failed.

Public entry point:
    parse_general_ledger(file_path: str, target_sheet: str = None) -> list

Dependencies:
    pandas, openpyxl
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("audit_agent.ledger_parser")
if not logger.handlers:
    # Library-friendly default: emit to stderr but let the host app reconfigure.
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration: the internal schema and the column aliases we accept.
# ---------------------------------------------------------------------------
# Each canonical field maps to the set of *normalized* header variations we
# will accept from the wild. Normalization (see _normalize_header) lower-cases
# the header, strips whitespace/punctuation, and collapses separators to a
# single underscore, so 'Transaction Date', 'transaction-date' and
# 'TRANSACTION_DATE' all reduce to 'transaction_date'.
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "transaction_date", "tx_date", "posting_date"],
    "vendor": ["vendor", "supplier", "payee", "vendor_name", "description"],
    "amount": ["amount", "amount_usd", "total", "debit", "value"],
    "invoice_no": ["invoice_no", "invoice_number", "inv_no", "reference", "ref_no"],
    # Optional: the GL account head a line was booked to (for the GL consistency
    # check). Not required — absent simply yields GL_UNDETERMINED downstream.
    "gl_account": [
        "gl_account", "account", "gl_head", "account_head", "gl",
        "account_name", "ledger_account", "gl_code", "gl_account_head",
    ],
}

# Columns the comparison engine cannot run without. If we can't locate these,
# we fail loudly so the UI can tell the user precisely what was missing.
REQUIRED_FIELDS: tuple[str, ...] = ("amount", "vendor")

# Offset applied to a pandas (0-based) row index to recover the human-facing
# Excel row number: +1 because Excel rows are 1-based, +1 more because the
# header occupies physical row 1. Hence pandas index 0 -> Excel row 2.
EXCEL_ROW_OFFSET: int = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_header(raw: Any) -> str:
    """
    Reduce an arbitrary, messy column header to a canonical comparison key.

    The normalization is intentionally aggressive so that cosmetic differences
    never cause a match to fail:

        "  Vendor Name "   -> "vendor_name"
        "Transaction-Date" -> "transaction_date"
        "Amount (USD)"     -> "amount_usd"
        "Ref. No."         -> "ref_no"

    Args:
        raw: The original header value (may be a non-string, e.g. NaN/int).

    Returns:
        A lower-cased, underscore-separated token string. Returns an empty
        string for blank/None headers.
    """
    if raw is None:
        return ""

    text = str(raw).strip().lower()
    if not text:
        return ""

    # Replace any run of non-alphanumeric characters with a single underscore.
    text = re.sub(r"[^a-z0-9]+", "_", text)
    # Collapse duplicate underscores and trim leading/trailing ones.
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _match_field(normalized: str, alias_to_field: dict[str, str]) -> str | None:
    """
    Resolve a single normalized header to a canonical field name.

    Matching happens in two tiers, strongest first, so we never let a fuzzy
    guess override an exact hit:

        1. Exact match: the normalized header equals a known alias outright
           (e.g. "vendor_name" -> 'vendor').
        2. Token-subset match: every token of some alias appears in the
           header's token set (e.g. "supplier_name" contains the alias
           "supplier" -> 'vendor'; "amount_in_usd" contains "amount").
           This is what makes real exports ("Supplier Name", "Total Amount
           (USD)") match without having to enumerate every permutation.

    To keep tier 2 safe, longer (more specific) aliases are tried first, and a
    single-token alias must appear as a whole word in the header — never as a
    substring of another token — so "value" never accidentally claims "valuation".

    Args:
        normalized: The normalized header token string to classify.
        alias_to_field: Inverted {normalized_alias: canonical_field} lookup.

    Returns:
        The canonical field name, or None if nothing matched.
    """
    # Tier 1 — exact.
    if normalized in alias_to_field:
        return alias_to_field[normalized]

    # Tier 2 — token-subset. Try the most specific (multi-token) aliases first.
    header_tokens = set(normalized.split("_"))
    for alias in sorted(alias_to_field, key=lambda a: a.count("_"), reverse=True):
        alias_tokens = set(alias.split("_"))
        if alias_tokens <= header_tokens:
            return alias_to_field[alias]

    return None


def _build_header_map(columns: list[Any]) -> dict[str, str]:
    """
    Map each *original* DataFrame column to its canonical internal field name.

    For every incoming column we compute its normalized form and resolve it via
    :func:`_match_field` (exact match preferred, token-subset fallback). The
    first canonical field that claims a given column wins, and a canonical field
    is only assigned once (first match by sheet order), so duplicate/ambiguous
    columns don't silently overwrite each other.

    Args:
        columns: The raw column labels as read from the worksheet.

    Returns:
        A dict mapping {original_column_label: canonical_field_name} for every
        column we were able to confidently classify. Unmatched columns are
        simply omitted.
    """
    # Invert COLUMN_ALIASES into {normalized_alias: canonical_field} for lookup.
    alias_to_field: dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_to_field[_normalize_header(alias)] = field

    header_map: dict[str, str] = {}
    claimed_fields: set[str] = set()

    for original in columns:
        normalized = _normalize_header(original)
        if not normalized:
            continue

        field = _match_field(normalized, alias_to_field)
        if field is None:
            logger.debug("Unmatched ledger column ignored: %r", original)
            continue

        if field in claimed_fields:
            # A second column wants the same canonical slot — keep the first,
            # warn so the auditor can investigate the source export.
            logger.warning(
                "Column %r also maps to '%s', which is already taken; ignoring it.",
                original,
                field,
            )
            continue

        header_map[original] = field
        claimed_fields.add(field)
        logger.debug("Mapped column %r -> '%s'", original, field)

    return header_map


def _clean_amount(value: Any) -> float | None:
    """
    Coerce a raw cell value into a deterministic float for mathematical safety.

    Handles the typical mess found in exported ledgers:
        - currency symbols and thousands separators: "$1,250.00" -> 1250.0
        - accounting-style negatives in parentheses: "(500.00)"  -> -500.0
        - trailing-minus formats:                    "500-"      -> -500.0
        - blank / dash placeholders:                 "", "-"     -> None

    Args:
        value: The raw amount cell.

    Returns:
        A float rounded to cents, or None if the value is empty/non-numeric.
        Rounding here guarantees amounts never carry binary float noise (e.g.
        921.0700000000001) into the downstream deterministic comparison.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # Fast path: already numeric.
    if isinstance(value, (int, float)):
        return round(float(value), 2)

    text = str(value).strip()
    if text in ("", "-", "—", "n/a", "na", "none"):
        return None

    negative = False

    # Accounting parentheses denote a negative number.
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    # Trailing minus (some ERP exports do "1,200.00-").
    if text.endswith("-"):
        negative = True
        text = text[:-1]

    # Strip everything that isn't a digit, decimal point, or leading minus.
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in ("", "-", "."):
        return None

    try:
        amount = float(text)
    except ValueError:
        return None

    amount = -abs(amount) if negative else amount
    return round(amount, 2)


def _clean_reference(value: Any) -> str:
    """
    Coerce a raw cell into a clean reference string (vendor name, invoice no.).

    Removes surrounding whitespace and normalizes the float artifacts pandas
    introduces (e.g. an invoice number read as 1001.0 -> "1001"). Empty/NaN
    cells become an empty string rather than the literal "nan".

    Args:
        value: The raw cell value.

    Returns:
        A trimmed string ("" when the cell is empty).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    # A whole-number float almost always means an ID that pandas widened;
    # render it without the spurious ".0".
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_general_ledger(file_path: str, target_sheet: str = None) -> list:
    """
    Parse a client General Ledger Excel file into a clean list of records.

    This is the single public entry point for Module 1. It reads the workbook,
    standardizes its columns into the internal schema, coerces values to safe
    types, drops empty spacer rows, and tags every record with the physical
    Excel row number it came from.

    Args:
        file_path:
            Absolute or relative path to the .xlsx workbook to parse.
        target_sheet:
            Optional worksheet name to read. If None, the first sheet in the
            workbook is used.

    Returns:
        A list of dictionaries, one per ledger line, each shaped like::

            {
                "ledger_row_index": 2,        # physical Excel row number
                "date": "2026-01-15",         # cleaned string (or "")
                "vendor": "Acme Corp",        # cleaned string
                "amount": 1250.0,             # float, math-safe
                "invoice_no": "INV-001",      # cleaned string (or "")
            }

        Canonical fields that were not present in the source are still included
        as keys with safe empty defaults so downstream code never KeyErrors on
        an optional field.

    Raises:
        FileNotFoundError:
            The workbook path does not exist.
        ValueError:
            The requested ``target_sheet`` is not in the workbook, or the file
            cannot be read as a valid Excel workbook.
        KeyError:
            A required column ('amount' and/or 'vendor') could not be matched
            from the source headers. The message names exactly what is missing.
    """
    # --- 1. Read the workbook -------------------------------------------------
    try:
        # sheet_name=0 selects the first sheet when the caller didn't specify one.
        sheet = target_sheet if target_sheet is not None else 0
        df = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
    except FileNotFoundError:
        logger.error("Ledger file not found: %s", file_path)
        raise FileNotFoundError(f"Ledger file not found: {file_path}")
    except ValueError as exc:
        # pandas raises ValueError for a missing/invalid sheet name.
        logger.error("Could not read sheet from %s: %s", file_path, exc)
        raise ValueError(
            f"Could not read worksheet '{target_sheet}' from '{file_path}': {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any reader failure cleanly.
        logger.error("Failed to open workbook %s: %s", file_path, exc)
        raise ValueError(
            f"'{file_path}' could not be read as a valid Excel workbook: {exc}"
        ) from exc

    logger.info(
        "Read %d raw rows and %d columns from %s.",
        len(df),
        len(df.columns),
        file_path,
    )

    # --- 2. Drop entirely empty spacer rows ----------------------------------
    # Many exports interleave blank separator rows; `how="all"` removes only the
    # rows where *every* cell is NaN, preserving the original index values so we
    # can still recover the true Excel line number below.
    df = df.dropna(how="all")

    # --- 3. Resolve the column mapping ---------------------------------------
    header_map = _build_header_map(list(df.columns))
    matched_fields = set(header_map.values())

    # Fail loudly if any critical column is missing.
    missing_required = [f for f in REQUIRED_FIELDS if f not in matched_fields]
    if missing_required:
        # Help the user by listing which headers we *did* see.
        seen = ", ".join(str(c) for c in df.columns) or "<none>"
        raise KeyError(
            "Cannot parse General Ledger: required column(s) "
            f"{missing_required} could not be matched. "
            f"Columns found in the sheet were: [{seen}]. "
            "Please rename the relevant column(s) or extend the alias table."
        )

    # Rename matched columns to their canonical names; keep only those columns.
    df = df.rename(columns=header_map)
    canonical_columns = [c for c in COLUMN_ALIASES if c in df.columns]
    df = df[canonical_columns]

    # --- 4. Build clean records ----------------------------------------------
    records: list[dict] = []
    for idx, row in df.iterrows():
        try:
            amount = _clean_amount(row.get("amount"))
            vendor = _clean_reference(row.get("vendor"))

            # A row with neither a usable amount nor a vendor is noise — skip it.
            if amount is None and not vendor:
                logger.debug("Skipping empty/non-data row at index %s.", idx)
                continue

            record = {
                # Physical Excel line number: pandas index + header + 1-based offset.
                "ledger_row_index": int(idx) + EXCEL_ROW_OFFSET,
                "date": _clean_reference(row.get("date")),
                "vendor": vendor,
                "amount": amount,
                "invoice_no": _clean_reference(row.get("invoice_no")),
                # Optional GL account head (empty string if the column is absent).
                "gl_account": _clean_reference(row.get("gl_account")),
            }
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - never let one bad row abort the run.
            logger.warning(
                "Failed to parse ledger row at Excel line ~%s: %s",
                int(idx) + EXCEL_ROW_OFFSET,
                exc,
            )
            continue

    logger.info("Parsed %d clean ledger records.", len(records))
    return records


# ---------------------------------------------------------------------------
# Isolated self-test: prove the module runs end-to-end with dummy data.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import tempfile

    logging.getLogger("audit_agent.ledger_parser").setLevel(logging.INFO)

    # Build a deliberately messy, SAP/QuickBooks-style ledger in memory.
    dummy = pd.DataFrame(
        {
            "Posting Date": ["2026-01-15", "2026-01-16", None, "2026-01-18"],
            "Supplier Name": ["Acme Corp", "Globex LLC", None, "Initech"],
            "Amount (USD)": ["$1,250.00", "(500.00)", None, "3400"],
            "Inv. No.": ["INV-001", "INV-002", None, 1004],
            "Notes": ["ok", "credit memo", None, "rush"],  # unmatched column, ignored
        }
    )

    tmp_dir = tempfile.mkdtemp(prefix="audit_agent_")
    sample_path = os.path.join(tmp_dir, "dummy_ledger.xlsx")
    dummy.to_excel(sample_path, index=False, engine="openpyxl")

    print(f"\n[self-test] Wrote dummy ledger to: {sample_path}\n")

    parsed = parse_general_ledger(sample_path)

    print(f"[self-test] parse_general_ledger returned {len(parsed)} record(s):\n")
    for rec in parsed:
        print(f"  {rec}")

    # --- Lightweight assertions to prove correctness -------------------------
    assert len(parsed) == 3, "Expected 3 data rows (the blank spacer row is dropped)."
    assert parsed[0]["ledger_row_index"] == 2, "First data row must map to Excel row 2."
    assert parsed[0]["amount"] == 1250.0, "Currency symbols/commas must be stripped."
    assert parsed[1]["amount"] == -500.0, "Parentheses must yield a negative amount."
    assert parsed[1]["ledger_row_index"] == 3, "Second data row must map to Excel row 3."
    # The blank row was physical Excel row 4; the next real row is Excel row 5.
    assert parsed[2]["ledger_row_index"] == 5, "Row index must survive the dropped spacer."
    assert parsed[2]["invoice_no"] == "1004", "Numeric invoice IDs must render without '.0'."

    # --- Prove required-column enforcement -----------------------------------
    bad = pd.DataFrame({"Posting Date": ["2026-01-15"], "Memo": ["no money column"]})
    bad_path = os.path.join(tmp_dir, "bad_ledger.xlsx")
    bad.to_excel(bad_path, index=False, engine="openpyxl")
    try:
        parse_general_ledger(bad_path)
        raise AssertionError("Expected a KeyError for the missing 'amount'/'vendor' columns.")
    except KeyError as exc:
        print(f"\n[self-test] Correctly raised KeyError for missing columns:\n  {exc}\n")

    print("[self-test] All assertions passed. Module is healthy. [OK]")
