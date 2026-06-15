"""
Audit Agent: Ledger-to-Invoice Vouching Engine
Module 3 — The Reconciliation / Match Engine
============================================

This module performs the strict, *deterministic* reconciliation at the heart of
the system. It cross-references every standardized General Ledger transaction
(produced by Module 1, ``ledger_parser``) against the structured invoice
metadata extracted from source documents (produced by Module 2,
``document_ocr``).

Design philosophy
-----------------
Fuzzy / probabilistic logic is used ONLY to *find* the candidate document a
ledger line is most likely referring to (because human bookkeepers misspell
vendor names). Once a candidate is found, the financial verdict is rendered by
pure, deterministic arithmetic:

    variance = round(ledger_amount - invoice_amount, 2)

Determinism does **not** mean zero tolerance. Real ledgers legitimately differ
from invoices by small amounts (rounding, sales tax, FX). The engine therefore
applies a *configurable, fully reproducible* tolerance band:

    allowed = max(absolute_tolerance, |invoice_amount| * relative_tolerance_pct)

A line within that band is VERIFIED_WITHIN_TOLERANCE (with the exact variance
logged); anything beyond it is an EXCEPTION. The same inputs always yield the
same verdict — there is no AI guessing on numbers.

Statuses
--------
Ledger-driven (existence assertion):
    - "VERIFIED"                  amount ties exactly + invoice no. + vendor match.
    - "VERIFIED_WITHIN_TOLERANCE" matches, amount within the tolerance band.
    - "EXCEPTION"                 a doc was found but it doesn't tie out.
    - "MISSING_DOC"               no document could be matched.
    - "POTENTIAL_DUPLICATE_CLAIM" a later ledger line claims an invoice already
                                  claimed by an earlier line (possible double-book).

Invoice-driven (completeness assertion / tooling):
    - "UNRECORDED_INVOICE"        an uploaded invoice matched no ledger line
                                  (potential unrecorded liability).
    - "PROCESSING_ERROR"          an invoice could not be read/extracted by the
                                  OCR/AI stage — a TOOLING failure, explicitly
                                  NOT a missing-document audit finding.

Public entry point:
    reconcile_ledger_with_invoices(
        ledger_data, invoice_data,
        absolute_tolerance=0.05, relative_tolerance_pct=0.005,
    ) -> dict

Dependencies:
    thefuzz   (recommended; the module degrades gracefully if it is absent)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Fuzzy matching dependency (used only to LOCATE candidates, never to judge money).
# ---------------------------------------------------------------------------
try:
    from thefuzz import fuzz

    def _vendor_similarity(a: str, b: str) -> int:
        """
        Return a 0-100 vendor similarity score blending three thefuzz scorers.

        Real ledgers abbreviate corporate suffixes ("Acme Corp" vs "Acme
        Corporation"). Token scorers alone under-score these, so we also factor
        in ``partial_ratio`` (lightly discounted to curb substring over-matching).
        Safe to be slightly generous: vendor similarity is only ONE of three
        gates for a VERIFIED verdict, so it can never single-handedly create a
        false match.
        """
        return max(
            fuzz.token_set_ratio(a, b),
            fuzz.token_sort_ratio(a, b),
            fuzz.partial_ratio(a, b) - 5,
        )

except ImportError:  # pragma: no cover - fallback path
    from difflib import SequenceMatcher

    def _vendor_similarity(a: str, b: str) -> int:
        """Return a 0-100 similarity score using stdlib difflib (fallback)."""
        return int(round(SequenceMatcher(None, a, b).ratio() * 100))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("audit_agent.match_engine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Status constants (single source of truth — imported by the report generator).
# ---------------------------------------------------------------------------
STATUS_VERIFIED = "VERIFIED"
STATUS_VERIFIED_TOL = "VERIFIED_WITHIN_TOLERANCE"
STATUS_EXCEPTION = "EXCEPTION"
STATUS_MISSING = "MISSING_DOC"
STATUS_DUPLICATE = "POTENTIAL_DUPLICATE_CLAIM"
STATUS_UNRECORDED = "UNRECORDED_INVOICE"
STATUS_PROCESSING_ERROR = "PROCESSING_ERROR"


# ---------------------------------------------------------------------------
# Tunable thresholds (deterministic, documented, version-controlled).
# ---------------------------------------------------------------------------
VENDOR_MATCH_THRESHOLD: int = 88          # vendor similarity for a clean match
VENDOR_CANDIDATE_THRESHOLD: int = 70      # vendor similarity to even be a candidate
MONEY_PRECISION: int = 2                  # money compared to the penny

# Default reconciliation tolerances (callers/UI may override per run).
DEFAULT_ABSOLUTE_TOLERANCE: float = 0.05      # $0.05 — rounding / minor pennies
DEFAULT_RELATIVE_TOLERANCE_PCT: float = 0.005  # 0.5%  — small local tax variances

# Keys probed (in order) to discover a human-facing file reference on an invoice.
_FILE_REF_KEYS = (
    "matching_invoice_file",
    "source_file",
    "file_path",
    "file",
    "filename",
    "document",
    "source_document",
)

# Keys/values that indicate the OCR/AI stage failed to read an invoice.
_ERROR_FLAG_KEYS = ("processing_error", "extraction_error", "error", "failed")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _norm_invoice_no(value: Any) -> str:
    """Normalize an invoice number for strict equality (e.g. 'INV-001'->'INV001')."""
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _norm_vendor(value: Any) -> str:
    """Normalize a vendor name for fuzzy comparison (lowercase, de-punctuated)."""
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_money(value: Any) -> Optional[float]:
    """Coerce a value to a penny-rounded float, or None if not numeric."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int — guard it out
        return None
    if isinstance(value, (int, float)):
        return round(float(value), MONEY_PRECISION)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", ".", "-"):
        return None
    try:
        return round(float(cleaned), MONEY_PRECISION)
    except ValueError:
        return None


def _invoice_file_ref(invoice: dict) -> Optional[str]:
    """Best-effort human-facing reference for which document a match came from."""
    for key in _FILE_REF_KEYS:
        ref = invoice.get(key)
        if ref:
            return str(ref)
    inv_no = invoice.get("invoice_number")
    return str(inv_no) if inv_no else None


def _has_processing_error(invoice: dict) -> bool:
    """
    True if the OCR/AI stage flagged this invoice as unreadable/failed.

    Honors an explicit boolean flag from Module 2 (``processing_error``) and a
    few common aliases, plus a self-declared error status.
    """
    for key in _ERROR_FLAG_KEYS:
        if invoice.get(key):
            return True
    status = str(invoice.get("status", "")).upper()
    return status in (STATUS_PROCESSING_ERROR, "NEEDS_REVIEW")


def _invoice_identity(invoice: dict) -> tuple:
    """
    A stable identity for duplicate-claim tracking.

    Prefers the normalized invoice number (so two ledger lines pointing at the
    same invoice number collide), falling back to object identity when no
    invoice number is present.
    """
    inv_no = _norm_invoice_no(invoice.get("invoice_number"))
    return ("invno", inv_no) if inv_no else ("obj", id(invoice))


def _tolerance_allowed(
    reference_amount: Optional[float], abs_tol: float, rel_tol: float
) -> float:
    """
    Compute the deterministic allowed variance band for a comparison.

    allowed = max(absolute_tolerance, |reference_amount| * relative_tolerance_pct)

    The relative component scales with invoice size (a 0.5% tax rounding on a
    $100k invoice is larger than on a $100 one); the absolute floor covers small
    fixed rounding. Both are caller-supplied and fully reproducible.
    """
    rel = abs(reference_amount) * rel_tol if reference_amount is not None else 0.0
    return round(max(abs_tol, rel), MONEY_PRECISION)


# ---------------------------------------------------------------------------
# Candidate selection (the only place fuzzy logic is allowed)
# ---------------------------------------------------------------------------
def _score_candidate(ledger: dict, invoice: dict) -> tuple[int, dict]:
    """
    Score how strongly an invoice could be the document for a ledger line.

    Used ONLY to pick the best candidate to evaluate — never to decide the
    financial verdict. +1000 for an exact invoice-no match, plus vendor
    similarity, plus a small bonus when amounts already tie.
    """
    led_inv = _norm_invoice_no(ledger.get("ledger_invoice_no", ledger.get("invoice_no")))
    inv_inv = _norm_invoice_no(invoice.get("invoice_number"))

    led_vendor = _norm_vendor(ledger.get("ledger_vendor", ledger.get("vendor")))
    inv_vendor = _norm_vendor(invoice.get("vendor_name"))

    led_amount = _to_money(ledger.get("ledger_amount", ledger.get("amount")))
    inv_amount = _to_money(invoice.get("total_amount"))

    invoice_no_match = bool(led_inv) and led_inv == inv_inv
    vendor_score = (
        _vendor_similarity(led_vendor, inv_vendor) if led_vendor and inv_vendor else 0
    )
    amounts_tie = (
        led_amount is not None
        and inv_amount is not None
        and round(led_amount - inv_amount, MONEY_PRECISION) == 0.0
    )

    score = 0
    if invoice_no_match:
        score += 1000
    score += vendor_score
    if amounts_tie:
        score += 50

    detail = {
        "invoice_no_match": invoice_no_match,
        "vendor_score": vendor_score,
        "led_amount": led_amount,
        "inv_amount": inv_amount,
        "amounts_tie": amounts_tie,
    }
    return score, detail


def _find_best_candidate(
    ledger: dict, invoices: list
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Select the single best candidate invoice for a ledger line, if any qualifies.

    A candidate qualifies if the invoice number matches exactly OR the vendor
    similarity clears VENDOR_CANDIDATE_THRESHOLD. Highest score wins.
    """
    best_invoice: Optional[dict] = None
    best_detail: Optional[dict] = None
    best_score = -1

    for invoice in invoices:
        score, detail = _score_candidate(ledger, invoice)
        qualifies = detail["invoice_no_match"] or (
            detail["vendor_score"] >= VENDOR_CANDIDATE_THRESHOLD
        )
        if qualifies and score > best_score:
            best_score = score
            best_invoice = invoice
            best_detail = detail

    return best_invoice, best_detail


# ---------------------------------------------------------------------------
# Result-row builders
# ---------------------------------------------------------------------------
def _base_row(ledger: dict) -> dict:
    """Build the shared scaffold for a ledger-driven result row."""
    led_vendor = ledger.get("ledger_vendor", ledger.get("vendor")) or ""
    led_invoice_no = ledger.get("ledger_invoice_no", ledger.get("invoice_no")) or ""
    led_amount = _to_money(ledger.get("ledger_amount", ledger.get("amount")))
    return {
        "ledger_row_index": ledger.get("ledger_row_index"),
        "ledger_vendor": str(led_vendor),
        "ledger_amount": led_amount if led_amount is not None else 0.0,
        "ledger_invoice_no": str(led_invoice_no),
        "status": STATUS_MISSING,
        "variance": 0.0,
        "matching_invoice_file": None,
        "audit_notes": "",
    }


def _missing_result(ledger: dict, processing_error_count: int) -> dict:
    """Build a MISSING_DOC row, with a caveat when tooling errors occurred."""
    result = _base_row(ledger)
    led_amount = result["ledger_amount"]
    result["status"] = STATUS_MISSING
    # The whole ledger amount is unsupported; express it as the variance.
    result["variance"] = led_amount
    notes = (
        "No supporting invoice document could be matched to this ledger line "
        f"(searched on invoice no. '{result['ledger_invoice_no']}' and vendor "
        f"'{result['ledger_vendor']}'). The full amount is unverified."
    )
    if processing_error_count:
        notes += (
            f" NOTE: {processing_error_count} invoice(s) failed AI extraction this "
            "run — if one of them corresponds to this line, this is a tooling "
            "limitation, not a confirmed missing document. Verify manually."
        )
    result["audit_notes"] = notes
    return result


def _evaluate_matched_line(
    ledger: dict, invoice: dict, detail: dict, abs_tol: float, rel_tol: float
) -> dict:
    """
    Render the deterministic verdict for a ledger line with a matched candidate.

    Produces VERIFIED, VERIFIED_WITHIN_TOLERANCE, or EXCEPTION.
    """
    result = _base_row(ledger)
    result["matching_invoice_file"] = _invoice_file_ref(invoice)

    led_amount = detail["led_amount"]
    inv_amount = detail["inv_amount"]
    invoice_no_match = detail["invoice_no_match"]
    vendor_score = detail["vendor_score"]
    vendor_ok = vendor_score >= VENDOR_MATCH_THRESHOLD

    # --- Deterministic amount evaluation -------------------------------------
    if led_amount is None or inv_amount is None:
        variance = led_amount if led_amount is not None else 0.0
        result["variance"] = variance
        amount_exact = amount_within_tol = False
        allowed = 0.0
    else:
        variance = round(led_amount - inv_amount, MONEY_PRECISION)
        allowed = _tolerance_allowed(inv_amount, abs_tol, rel_tol)
        result["variance"] = variance
        amount_exact = variance == 0.0
        amount_within_tol = abs(variance) <= allowed

    info_ok = invoice_no_match and vendor_ok

    # --- VERIFIED (exact, all three gates) -----------------------------------
    if info_ok and amount_exact:
        result["status"] = STATUS_VERIFIED
        result["audit_notes"] = (
            "VERIFIED: invoice number matches exactly, vendor matches "
            f"(similarity {vendor_score}%), and the amount ties out to the penny "
            f"(variance {variance:.2f})."
        )
        return result

    # --- VERIFIED_WITHIN_TOLERANCE (info matches, amount inside the band) -----
    if info_ok and amount_within_tol:
        result["status"] = STATUS_VERIFIED_TOL
        result["audit_notes"] = (
            "VERIFIED WITHIN TOLERANCE: invoice number and vendor match; amount "
            f"variance of {variance:.2f} is within the allowed tolerance of "
            f"±{allowed:.2f} (abs ${abs_tol:.2f} / rel {rel_tol * 100:.3g}%). "
            "Likely rounding, tax, or FX. Logged for auditor awareness."
        )
        return result

    # --- EXCEPTION (anything else) -------------------------------------------
    reasons: list[str] = []
    if led_amount is None or inv_amount is None:
        if inv_amount is None:
            reasons.append("the invoice has no readable total amount to compare")
        else:
            reasons.append("the ledger line has no readable amount to compare")
    elif not amount_within_tol:
        reasons.append(
            f"amount variance of {variance:.2f} exceeds the allowed tolerance of "
            f"±{allowed:.2f} (ledger {led_amount:.2f} vs invoice {inv_amount:.2f})"
        )

    if not invoice_no_match:
        reasons.append(
            f"invoice number mismatch (ledger '{result['ledger_invoice_no']}' vs "
            f"document '{invoice.get('invoice_number')}')"
        )
    if not vendor_ok:
        reasons.append(
            f"vendor only a weak match (similarity {vendor_score}% < "
            f"{VENDOR_MATCH_THRESHOLD}% threshold; ledger '{result['ledger_vendor']}' "
            f"vs document '{invoice.get('vendor_name')}')"
        )

    result["status"] = STATUS_EXCEPTION
    result["audit_notes"] = (
        "EXCEPTION: a candidate document was found but it does not fully tie out. "
        "Issues: " + "; ".join(reasons) + "."
    )
    return result


def _duplicate_result(
    ledger: dict, invoice: dict, detail: dict, first_row: Any
) -> dict:
    """Build a POTENTIAL_DUPLICATE_CLAIM row referencing the prior claimant."""
    result = _base_row(ledger)
    result["matching_invoice_file"] = _invoice_file_ref(invoice)

    led_amount = detail["led_amount"]
    inv_amount = detail["inv_amount"]
    variance = (
        round(led_amount - inv_amount, MONEY_PRECISION)
        if led_amount is not None and inv_amount is not None
        else (led_amount if led_amount is not None else 0.0)
    )
    result["variance"] = variance
    result["status"] = STATUS_DUPLICATE
    result["audit_notes"] = (
        "POTENTIAL DUPLICATE CLAIM: this ledger line matches invoice "
        f"'{invoice.get('invoice_number')}' "
        f"({_invoice_file_ref(invoice)}), which was ALREADY claimed by ledger "
        f"row {first_row}. Possible duplicate booking/payment — requires auditor "
        "investigation."
    )
    return result


def _unrecorded_result(invoice: dict) -> dict:
    """Build an UNRECORDED_INVOICE row for an invoice that matched no ledger line."""
    inv_amount = _to_money(invoice.get("total_amount"))
    return {
        "ledger_row_index": None,
        "ledger_vendor": str(invoice.get("vendor_name") or ""),
        "ledger_amount": inv_amount if inv_amount is not None else 0.0,
        "ledger_invoice_no": str(invoice.get("invoice_number") or ""),
        "status": STATUS_UNRECORDED,
        "variance": inv_amount if inv_amount is not None else 0.0,
        "matching_invoice_file": _invoice_file_ref(invoice),
        "audit_notes": (
            "COMPLETENESS RISK: this invoice was uploaded but could not be matched "
            "to any ledger line — a potential unrecorded liability / unbooked "
            "expense. The full amount is shown as the exposure. Requires auditor "
            "validation."
        ),
    }


def _processing_error_result(invoice: dict) -> dict:
    """Build a PROCESSING_ERROR row for an invoice the OCR/AI stage couldn't read."""
    inv_amount = _to_money(invoice.get("total_amount"))
    detail = invoice.get("error_detail") or invoice.get("error") or "unknown error"
    return {
        "ledger_row_index": None,
        "ledger_vendor": str(invoice.get("vendor_name") or "(unreadable)"),
        "ledger_amount": inv_amount if inv_amount is not None else 0.0,
        "ledger_invoice_no": str(invoice.get("invoice_number") or ""),
        "status": STATUS_PROCESSING_ERROR,
        "variance": 0.0,
        "matching_invoice_file": _invoice_file_ref(invoice),
        "audit_notes": (
            f"AI EXTRACTION FAILURE ({detail}). This document could not be read or "
            "parsed by the OCR/AI stage and was therefore NOT included in matching. "
            "This is a TOOLING limitation, NOT a missing-document audit finding — "
            "re-upload a clearer copy or review the source manually."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reconcile_ledger_with_invoices(
    ledger_data: list,
    invoice_data: list,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance_pct: float = DEFAULT_RELATIVE_TOLERANCE_PCT,
) -> dict:
    """
    Reconcile ledger transactions against extracted invoice metadata.

    Performs a bidirectional reconciliation:
      * Pass 1 (existence): each ledger line is matched to its best candidate
        invoice and given a deterministic verdict within the tolerance band,
        with duplicate-claim detection.
      * Pass 2 (completeness): any valid invoice never claimed by a ledger line
        is surfaced as a potential unrecorded liability.
      * Pass 3 (tooling): invoices the OCR/AI stage flagged as unreadable are
        surfaced as processing errors — never disguised as missing documents.

    Args:
        ledger_data:            Standardized ledger dicts from Module 1.
        invoice_data:           Extracted-invoice dicts from Module 2 (may carry a
                                ``processing_error`` flag and a ``source_file``).
        absolute_tolerance:     Allowed absolute variance in dollars (default 0.05).
        relative_tolerance_pct: Allowed variance as a fraction of the invoice
                                amount (default 0.005 == 0.5%).

    Returns:
        A structured payload: ``{"summary": {...}, "results": [...]}``. The summary
        carries a count for every status plus the tolerance settings used.

    Raises:
        TypeError: If either argument is not a list.
    """
    if not isinstance(ledger_data, list) or not isinstance(invoice_data, list):
        raise TypeError("Both ledger_data and invoice_data must be lists.")

    invoice_data = invoice_data or []

    # Split invoices into ones we could read vs ones the OCR/AI stage failed on.
    valid_invoices = [
        inv for inv in invoice_data if isinstance(inv, dict) and not _has_processing_error(inv)
    ]
    errored_invoices = [
        inv for inv in invoice_data if isinstance(inv, dict) and _has_processing_error(inv)
    ]
    processing_error_count = len(errored_invoices)

    results: list[dict] = []
    claimed_by: dict[tuple, Any] = {}      # invoice identity -> first claiming row
    matched_invoice_ids: set[int] = set()  # id() of every invoice claimed at all

    # --- Pass 1: ledger -> invoice (existence) -------------------------------
    for ledger in ledger_data:
        if not isinstance(ledger, dict):
            logger.warning("Skipping non-dict ledger entry: %r", ledger)
            continue

        try:
            best_invoice, detail = _find_best_candidate(ledger, valid_invoices)

            if best_invoice is None or detail is None:
                result = _missing_result(ledger, processing_error_count)
            else:
                matched_invoice_ids.add(id(best_invoice))
                identity = _invoice_identity(best_invoice)
                if identity in claimed_by:
                    result = _duplicate_result(
                        ledger, best_invoice, detail, claimed_by[identity]
                    )
                else:
                    claimed_by[identity] = ledger.get("ledger_row_index")
                    result = _evaluate_matched_line(
                        ledger, best_invoice, detail,
                        absolute_tolerance, relative_tolerance_pct,
                    )
        except Exception as exc:  # noqa: BLE001 - never abort the whole batch
            logger.error(
                "Failed to reconcile ledger row %r: %s",
                ledger.get("ledger_row_index"), exc,
            )
            result = _base_row(ledger)
            result["status"] = STATUS_EXCEPTION
            result["audit_notes"] = (
                f"EXCEPTION: internal error while reconciling this line: {exc}"
            )

        results.append(result)

    ledger_result_count = len(results)

    # --- Pass 2: invoice -> ledger (completeness / unrecorded liabilities) ---
    for invoice in valid_invoices:
        if id(invoice) not in matched_invoice_ids:
            results.append(_unrecorded_result(invoice))

    # --- Pass 3: surface tooling failures distinctly -------------------------
    for invoice in errored_invoices:
        results.append(_processing_error_result(invoice))

    # --- Tally -------------------------------------------------------------
    counts = {
        STATUS_VERIFIED: 0,
        STATUS_VERIFIED_TOL: 0,
        STATUS_EXCEPTION: 0,
        STATUS_MISSING: 0,
        STATUS_DUPLICATE: 0,
        STATUS_UNRECORDED: 0,
        STATUS_PROCESSING_ERROR: 0,
    }
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    summary = {
        "total_ledger_records": ledger_result_count,
        "total_invoices": len(invoice_data),
        "verified_count": counts[STATUS_VERIFIED],
        "verified_within_tolerance_count": counts[STATUS_VERIFIED_TOL],
        "exception_count": counts[STATUS_EXCEPTION],
        "missing_doc_count": counts[STATUS_MISSING],
        "potential_duplicate_count": counts[STATUS_DUPLICATE],
        "unrecorded_invoice_count": counts[STATUS_UNRECORDED],
        "processing_error_count": counts[STATUS_PROCESSING_ERROR],
        "tolerance_absolute": round(absolute_tolerance, MONEY_PRECISION),
        "tolerance_relative_pct": relative_tolerance_pct,
    }

    payload = {"summary": summary, "results": results}

    logger.info(
        "Reconciliation complete: %d ledger lines | verified=%d, within_tol=%d, "
        "exceptions=%d, missing=%d, duplicates=%d, unrecorded=%d, errors=%d.",
        ledger_result_count,
        counts[STATUS_VERIFIED], counts[STATUS_VERIFIED_TOL],
        counts[STATUS_EXCEPTION], counts[STATUS_MISSING],
        counts[STATUS_DUPLICATE], counts[STATUS_UNRECORDED],
        counts[STATUS_PROCESSING_ERROR],
    )
    return payload


# ---------------------------------------------------------------------------
# Isolated self-test: exercises every status path.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.getLogger("audit_agent.match_engine").setLevel(logging.INFO)

    mock_ledger = [
        # 1) Perfect exact match (fuzzy vendor) -> VERIFIED
        {"ledger_row_index": 2, "ledger_vendor": "Acme Corp",
         "ledger_amount": 1250.00, "ledger_invoice_no": "INV-001"},
        # 2) $0.03 off, within tolerance -> VERIFIED_WITHIN_TOLERANCE
        {"ledger_row_index": 3, "ledger_vendor": "Globex LLC",
         "ledger_amount": 500.03, "ledger_invoice_no": "INV-002"},
        # 3) No document at all -> MISSING_DOC
        {"ledger_row_index": 4, "ledger_vendor": "Initech",
         "ledger_amount": 3400.00, "ledger_invoice_no": "INV-404"},
        # 4) $150 off, beyond tolerance -> EXCEPTION
        {"ledger_row_index": 5, "ledger_vendor": "Wayne Ent",
         "ledger_amount": 750.00, "ledger_invoice_no": "INV-003"},
        # 5) Claims INV-001 again -> POTENTIAL_DUPLICATE_CLAIM (refers to row 2)
        {"ledger_row_index": 6, "ledger_vendor": "Acme Corp",
         "ledger_amount": 1250.00, "ledger_invoice_no": "INV-001"},
    ]

    mock_invoices = [
        {"vendor_name": "Acme Corporation", "invoice_number": "INV-001",
         "date": "2026-01-15", "total_amount": 1250.00, "source_file": "acme_001.pdf"},
        {"vendor_name": "Globex LLC", "invoice_number": "INV-002",
         "date": "2026-01-16", "total_amount": 500.00, "source_file": "globex_002.pdf"},
        {"vendor_name": "Wayne Enterprises", "invoice_number": "INV-003",
         "date": "2026-01-17", "total_amount": 900.00, "source_file": "wayne_003.pdf"},
        # Never matched by any ledger line -> UNRECORDED_INVOICE
        {"vendor_name": "Umbrella Inc", "invoice_number": "INV-777",
         "date": "2026-01-20", "total_amount": 9999.99, "source_file": "umbrella_777.pdf"},
        # Flagged by OCR as unreadable -> PROCESSING_ERROR (NOT missing doc)
        {"vendor_name": None, "invoice_number": None, "date": None,
         "total_amount": None, "source_file": "blurry_scan.pdf",
         "processing_error": True, "error_detail": "Groq API rate limit (HTTP 429)"},
    ]

    report = reconcile_ledger_with_invoices(mock_ledger, mock_invoices)

    print("\n[self-test] Reconciliation report:\n")
    print(json.dumps(report, indent=2))

    s = report["summary"]
    assert s["total_ledger_records"] == 5
    assert s["verified_count"] == 1, "one exact VERIFIED"
    assert s["verified_within_tolerance_count"] == 1, "one VERIFIED_WITHIN_TOLERANCE"
    assert s["exception_count"] == 1, "one EXCEPTION"
    assert s["missing_doc_count"] == 1, "one MISSING_DOC"
    assert s["potential_duplicate_count"] == 1, "one POTENTIAL_DUPLICATE_CLAIM"
    assert s["unrecorded_invoice_count"] == 1, "one UNRECORDED_INVOICE"
    assert s["processing_error_count"] == 1, "one PROCESSING_ERROR"

    by_row = {r["ledger_row_index"]: r for r in report["results"] if r["ledger_row_index"]}
    assert by_row[2]["status"] == "VERIFIED" and by_row[2]["variance"] == 0.0
    assert by_row[3]["status"] == "VERIFIED_WITHIN_TOLERANCE" and by_row[3]["variance"] == 0.03
    assert by_row[4]["status"] == "MISSING_DOC"
    assert by_row[5]["status"] == "EXCEPTION" and by_row[5]["variance"] == -150.0
    assert by_row[6]["status"] == "POTENTIAL_DUPLICATE_CLAIM"
    assert "row 2" in by_row[6]["audit_notes"], "duplicate must cite the first claimant"

    # The errored invoice must NOT have been treated as a missing document or
    # an unrecorded liability — it lives in its own PROCESSING_ERROR bucket.
    err_rows = [r for r in report["results"] if r["status"] == "PROCESSING_ERROR"]
    assert err_rows and "blurry_scan.pdf" in (err_rows[0]["matching_invoice_file"] or "")
    assert "AI EXTRACTION FAILURE" in err_rows[0]["audit_notes"]

    # Tolerance settings echoed back for the workpaper / reproducibility.
    assert s["tolerance_absolute"] == 0.05 and s["tolerance_relative_pct"] == 0.005

    print("\n[self-test] All assertions passed. Every status path verified. [OK]")
