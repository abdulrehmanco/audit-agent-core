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
STATUS_DUPLICATE = "POTENTIAL_DUPLICATE_CLAIM"   # same invoice on >1 LEDGER line (payment risk)
STATUS_DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"  # same invoice as >1 FILE, one ledger line (informational)
STATUS_UNRECORDED = "UNRECORDED_INVOICE"
STATUS_PROCESSING_ERROR = "PROCESSING_ERROR"


# ---------------------------------------------------------------------------
# Tunable thresholds (deterministic, documented, version-controlled).
# ---------------------------------------------------------------------------
VENDOR_MATCH_THRESHOLD: int = 88          # vendor similarity for a clean vendor-only match
# Vendor similarity required to even CONSIDER an invoice a candidate when the
# invoice number does NOT match. Kept high to avoid false matches like
# "LinkedIn" vs "Uline" (~75%) stealing an unrelated invoice.
VENDOR_CANDIDATE_THRESHOLD: int = 85
# NOTE: when the invoice NUMBER matches exactly AND the amount agrees, the verdict
# is VERIFIED regardless of vendor-name similarity (key precedence) — the vendor
# score never vetoes that, it's only recorded as an informational note. An
# acronym like "HP Inc" vs "Hewlett-Packard" (~29%) is plainly the same vendor.
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

    The invoice NUMBER is treated as the primary key (KEY PRECEDENCE): when it
    matches exactly AND the amount agrees, the verdict is VERIFIED regardless of
    the vendor-name score — the vendor never vetoes it (so "HP Inc" vs
    "Hewlett-Packard" verifies), with any vendor difference recorded as a note.
    When the match was made on vendor similarity alone (no invoice-number match),
    the stricter vendor threshold applies.

    Produces VERIFIED, VERIFIED_WITHIN_TOLERANCE, or EXCEPTION.
    """
    result = _base_row(ledger)
    result["matching_invoice_file"] = _invoice_file_ref(invoice)

    led_amount = detail["led_amount"]
    inv_amount = detail["inv_amount"]
    invoice_no_match = detail["invoice_no_match"]
    vendor_score = detail["vendor_score"]
    vendor_ok = vendor_score >= VENDOR_MATCH_THRESHOLD

    led_vendor = result["ledger_vendor"]
    doc_vendor = invoice.get("vendor_name")
    led_inv_no = result["ledger_invoice_no"]
    doc_inv_no = invoice.get("invoice_number")

    # --- Deterministic amount evaluation -------------------------------------
    if led_amount is None or inv_amount is None:
        variance = led_amount if led_amount is not None else 0.0
        amount_exact = amount_within_tol = False
        allowed = 0.0
    else:
        variance = round(led_amount - inv_amount, MONEY_PRECISION)
        allowed = _tolerance_allowed(inv_amount, abs_tol, rel_tol)
        amount_exact = variance == 0.0
        amount_within_tol = abs(variance) <= allowed
    result["variance"] = variance

    reasons: list[str] = []

    if invoice_no_match:
        # Invoice number identifies the document. Require only that the vendor is
        # plausibly the same entity; flag a markedly different vendor (which would
        # suggest a reused invoice number or a misposting).
        #
        # KEY PRECEDENCE: an exact invoice number + an agreeing amount is decisive
        # evidence on its own, so the vendor-name score NEVER vetoes a verify here
        # (an acronym like "HP Inc" vs "Hewlett-Packard" scores ~29% but is plainly
        # the same vendor). A vendor that isn't a clean match is recorded as an
        # informational note, not an exception.
        vendor_note = (
            ""
            if vendor_ok
            else (
                f" Note: vendor name differs (ledger '{led_vendor}' vs document "
                f"'{doc_vendor}', similarity {vendor_score}%); verified on the exact "
                "invoice number and amount."
            )
        )
        if amount_exact:
            result["status"] = STATUS_VERIFIED
            result["audit_notes"] = (
                "VERIFIED: invoice number matches exactly and the amount ties out "
                f"to the penny (variance {variance:.2f})." + vendor_note
            )
            return result
        elif amount_within_tol:
            result["status"] = STATUS_VERIFIED_TOL
            result["audit_notes"] = (
                "VERIFIED WITHIN TOLERANCE: invoice number matches; amount variance "
                f"of {variance:.2f} is within the allowed tolerance of ±{allowed:.2f} "
                f"(abs ${abs_tol:.2f} / rel {rel_tol * 100:.3g}%). Likely rounding, "
                "tax, or FX. Logged for auditor awareness." + vendor_note
            )
            return result
        else:
            # Invoice number matches but the amount is genuinely off — a real
            # discrepancy. Note any vendor difference too, but the amount drives it.
            reasons.append(
                f"amount variance of {variance:.2f} exceeds the allowed tolerance "
                f"of ±{allowed:.2f} (ledger {led_amount:.2f} vs invoice "
                f"{inv_amount:.2f})"
            )
            if not vendor_ok:
                reasons.append(
                    f"vendor name also differs (ledger '{led_vendor}' vs document "
                    f"'{doc_vendor}', similarity {vendor_score}%)"
                )
    else:
        # Matched on vendor similarity only — no invoice-number corroboration, so
        # apply the strict vendor threshold and flag any invoice-number conflict.
        if led_inv_no and doc_inv_no:
            reasons.append(
                f"invoice number mismatch (ledger '{led_inv_no}' vs document "
                f"'{doc_inv_no}')"
            )
        if not vendor_ok:
            reasons.append(
                f"vendor only a weak match (similarity {vendor_score}% < "
                f"{VENDOR_MATCH_THRESHOLD}% threshold; ledger '{led_vendor}' vs "
                f"document '{doc_vendor}')"
            )
        if led_amount is None or inv_amount is None:
            reasons.append("an amount is missing and cannot be compared")
        elif not amount_within_tol:
            reasons.append(
                f"amount variance of {variance:.2f} exceeds the allowed tolerance "
                f"of ±{allowed:.2f} (ledger {led_amount:.2f} vs invoice "
                f"{inv_amount:.2f})"
            )

        if not reasons:
            # Strong vendor, amounts tie, and the ledger carried no invoice number
            # to conflict with — accept it.
            if amount_exact:
                result["status"] = STATUS_VERIFIED
                result["audit_notes"] = (
                    f"VERIFIED: vendor matches (similarity {vendor_score}%) and the "
                    f"amount ties out to the penny (variance {variance:.2f}). "
                    "Matched without an invoice number."
                )
            else:
                result["status"] = STATUS_VERIFIED_TOL
                result["audit_notes"] = (
                    f"VERIFIED WITHIN TOLERANCE: vendor matches (similarity "
                    f"{vendor_score}%); amount variance of {variance:.2f} within "
                    f"±{allowed:.2f}. Matched without an invoice number."
                )
            return result

    # --- EXCEPTION (fell through) --------------------------------------------
    result["status"] = STATUS_EXCEPTION
    result["audit_notes"] = (
        "EXCEPTION: a candidate document was found but it does not fully tie out. "
        "Issues: " + "; ".join(reasons) + "."
    )
    return result


def _duplicate_document_result(invoice: dict) -> dict:
    """
    Build a DUPLICATE_DOCUMENT row for an unmatched invoice file whose number is
    the SAME as an invoice already matched to a (single) ledger line.

    This is a duplicate *document* (e.g. a re-scanned copy of an invoice that was
    already recorded against one ledger line) — NOT a duplicate payment claim and
    NOT an unrecorded liability. It is informational and is excluded from the
    "potential duplicate claims" tally.
    """
    inv_amount = _to_money(invoice.get("total_amount"))
    return {
        "ledger_row_index": None,
        "ledger_vendor": str(invoice.get("vendor_name") or ""),
        "ledger_amount": inv_amount if inv_amount is not None else 0.0,
        "ledger_invoice_no": str(invoice.get("invoice_number") or ""),
        "status": STATUS_DUPLICATE_DOCUMENT,
        "variance": 0.0,
        "matching_invoice_file": _invoice_file_ref(invoice),
        "audit_notes": (
            "DUPLICATE DOCUMENT (informational): this invoice file carries the same "
            f"invoice number ('{invoice.get('invoice_number')}') as an invoice "
            "already matched to a ledger line — it appears to be a redundant copy "
            "(e.g. a scan of an already-booked PDF), not an unrecorded liability "
            "and not a duplicate payment. No ledger impact; provided for awareness."
        ),
    }


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
    matched_invoice_ids: set[int] = set()          # id() of every invoice claimed at all
    claims_by_identity: dict[tuple, list[dict]] = {}  # invoice identity -> result rows

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
                result = _evaluate_matched_line(
                    ledger, best_invoice, detail,
                    absolute_tolerance, relative_tolerance_pct,
                )
                claims_by_identity.setdefault(identity, []).append(result)
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

    # --- Pass 1b: flag duplicate claims (an invoice claimed by >1 ledger line) -
    # Both/all postings are flagged so the auditor reviews every one of them.
    claimed_invoice_numbers: set[str] = set()
    for identity, rows in claims_by_identity.items():
        if identity[0] == "invno":
            claimed_invoice_numbers.add(identity[1])
        if len(rows) > 1:
            claim_rows = sorted(
                str(r["ledger_row_index"]) for r in rows if r["ledger_row_index"] is not None
            )
            inv_no = rows[0]["ledger_invoice_no"]
            for r in rows:
                r["status"] = STATUS_DUPLICATE
                r["audit_notes"] = (
                    f"POTENTIAL DUPLICATE CLAIM: invoice '{inv_no}' is referenced by "
                    f"{len(rows)} ledger lines (rows {', '.join(claim_rows)}). "
                    "Possible duplicate booking/payment — every posting requires "
                    "auditor investigation."
                )

    # --- Pass 2: invoice -> ledger (completeness / unrecorded liabilities) ---
    for invoice in valid_invoices:
        if id(invoice) in matched_invoice_ids:
            continue
        inv_no = _norm_invoice_no(invoice.get("invoice_number"))
        if inv_no and inv_no in claimed_invoice_numbers:
            # Same invoice number as one already matched -> duplicate document,
            # NOT an unrecorded liability.
            results.append(_duplicate_document_result(invoice))
        else:
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
        STATUS_DUPLICATE_DOCUMENT: 0,
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
        # Payment risk: same invoice on multiple LEDGER lines. Document copies
        # (same invoice as multiple FILES, one ledger line) are tracked separately
        # and intentionally NOT included here.
        "potential_duplicate_count": counts[STATUS_DUPLICATE],
        "duplicate_document_count": counts[STATUS_DUPLICATE_DOCUMENT],
        "unrecorded_invoice_count": counts[STATUS_UNRECORDED],
        "processing_error_count": counts[STATUS_PROCESSING_ERROR],
        "tolerance_absolute": round(absolute_tolerance, MONEY_PRECISION),
        "tolerance_relative_pct": relative_tolerance_pct,
    }

    payload = {"summary": summary, "results": results}

    logger.info(
        "Reconciliation complete: %d ledger lines | verified=%d, within_tol=%d, "
        "exceptions=%d, missing=%d, dup_claims=%d, dup_docs=%d, unrecorded=%d, "
        "errors=%d.",
        ledger_result_count,
        counts[STATUS_VERIFIED], counts[STATUS_VERIFIED_TOL],
        counts[STATUS_EXCEPTION], counts[STATUS_MISSING],
        counts[STATUS_DUPLICATE], counts[STATUS_DUPLICATE_DOCUMENT],
        counts[STATUS_UNRECORDED], counts[STATUS_PROCESSING_ERROR],
    )
    return payload


# ---------------------------------------------------------------------------
# Isolated self-test: exercises every status path.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.getLogger("audit_agent.match_engine").setLevel(logging.INFO)

    mock_ledger = [
        # 1) Exact match, fuzzy vendor ("Acme Corp" vs "Acme Corporation") -> VERIFIED
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
        # 5) Exact invoice no. + amount but weak vendor spelling -> VERIFIED
        #    (invoice number corroborates "Dell Corp" == "Dell Inc.")
        {"ledger_row_index": 6, "ledger_vendor": "Dell Corp",
         "ledger_amount": 4000.00, "ledger_invoice_no": "INV-9"},
        # 6 & 7) Same invoice claimed twice -> BOTH POTENTIAL_DUPLICATE_CLAIM
        {"ledger_row_index": 7, "ledger_vendor": "Beta LLC",
         "ledger_amount": 100.00, "ledger_invoice_no": "INV-777"},
        {"ledger_row_index": 8, "ledger_vendor": "Beta LLC",
         "ledger_amount": 100.00, "ledger_invoice_no": "INV-777"},
        # 8) Regression for the false-positive bug: a ledger line whose invoice
        #    genuinely doesn't exist must NOT latch onto an unrelated invoice via
        #    a weak (~75%) vendor near-miss -> MISSING_DOC.
        {"ledger_row_index": 9, "ledger_vendor": "LinkedIn",
         "ledger_amount": 8500.00, "ledger_invoice_no": "INV-LNK-60223"},
        # 9) The "victim" invoice's real ledger line must stay VERIFIED (not be
        #    demoted to a duplicate because something else claimed it first).
        {"ledger_row_index": 10, "ledger_vendor": "Uline",
         "ledger_amount": 4675.20, "ledger_invoice_no": "INV-UL-66902"},
        # 10) Penny-exact match from a float-noisy amount -> VERIFIED, never
        #     VERIFIED_WITHIN_TOLERANCE (regression for float rounding).
        {"ledger_row_index": 11, "ledger_vendor": "Comcast Business",
         "ledger_amount": 1240.6600000000001, "ledger_invoice_no": "INV-CMB-1"},
        # 11) Acronym vendor that will NEVER clear string similarity ("HP Inc" vs
        #     "Hewlett-Packard", ~29%) but has an exact invoice no. + amount ->
        #     KEY PRECEDENCE must return VERIFIED, not EXCEPTION.
        {"ledger_row_index": 12, "ledger_vendor": "HP Inc",
         "ledger_amount": 33450.00, "ledger_invoice_no": "INV-HP-90120"},
    ]

    mock_invoices = [
        {"vendor_name": "Acme Corporation", "invoice_number": "INV-001",
         "total_amount": 1250.00, "source_file": "acme_001.pdf"},
        {"vendor_name": "Globex LLC", "invoice_number": "INV-002",
         "total_amount": 500.00, "source_file": "globex_002.pdf"},
        {"vendor_name": "Wayne Enterprises", "invoice_number": "INV-003",
         "total_amount": 900.00, "source_file": "wayne_003.pdf"},
        {"vendor_name": "Dell Inc.", "invoice_number": "INV-9",
         "total_amount": 4000.00, "source_file": "dell_9.pdf"},
        {"vendor_name": "Beta LLC", "invoice_number": "INV-777",
         "total_amount": 100.00, "source_file": "beta_777.pdf"},
        # Never matched by any ledger line -> UNRECORDED_INVOICE
        {"vendor_name": "Umbrella Inc", "invoice_number": "INV-888",
         "total_amount": 9999.99, "source_file": "umbrella_888.pdf"},
        # Same invoice number as a recorded invoice -> duplicate DOCUMENT, not
        # an unrecorded liability -> POTENTIAL_DUPLICATE_CLAIM
        {"vendor_name": "Beta LLC", "invoice_number": "INV-777",
         "total_amount": 100.00, "source_file": "beta_777_SCAN.png"},
        # The unrelated invoice the "LinkedIn" line must NOT steal; matches its
        # own real ledger line (row 10) instead.
        {"vendor_name": "Uline", "invoice_number": "INV-UL-66902",
         "total_amount": 4675.20, "source_file": "uline_66902.pdf"},
        # Penny-exact partner for the float-rounding row (row 11).
        {"vendor_name": "Comcast Business", "invoice_number": "INV-CMB-1",
         "total_amount": 1240.66, "source_file": "comcast_1.pdf"},
        # Acronym partner for the key-precedence row (row 12) — vendor name is
        # totally different as a string but invoice no. + amount tie out exactly.
        {"vendor_name": "Hewlett-Packard", "invoice_number": "INV-HP-90120",
         "total_amount": 33450.00, "source_file": "hp_90120.pdf"},
        # Flagged by OCR as unreadable -> PROCESSING_ERROR (NOT missing doc)
        {"vendor_name": None, "invoice_number": None, "total_amount": None,
         "source_file": "blurry_scan.pdf",
         "processing_error": True, "error_detail": "Groq API rate limit (HTTP 429)"},
    ]

    report = reconcile_ledger_with_invoices(mock_ledger, mock_invoices)

    print("\n[self-test] Reconciliation report:\n")
    print(json.dumps(report, indent=2))

    s = report["summary"]
    assert s["total_ledger_records"] == 11
    assert s["verified_count"] == 5, "Acme, Dell, Uline, Comcast, HP(acronym)"
    assert s["verified_within_tolerance_count"] == 1, "one VERIFIED_WITHIN_TOLERANCE"
    assert s["exception_count"] == 1, "one EXCEPTION"
    assert s["missing_doc_count"] == 2, "Initech + LinkedIn (no false match)"
    # Payment-risk duplicate claims = the two INV-777 ledger postings ONLY.
    assert s["potential_duplicate_count"] == 2, "two ledger duplicate claims"
    # Document copy (scan of a recorded invoice) is separate and uncounted above.
    assert s["duplicate_document_count"] == 1, "one duplicate DOCUMENT (scan copy)"
    assert s["unrecorded_invoice_count"] == 1, "one UNRECORDED_INVOICE"
    assert s["processing_error_count"] == 1, "one PROCESSING_ERROR"

    by_row = {r["ledger_row_index"]: r for r in report["results"] if r["ledger_row_index"]}
    assert by_row[2]["status"] == "VERIFIED" and by_row[2]["variance"] == 0.0
    assert by_row[3]["status"] == "VERIFIED_WITHIN_TOLERANCE" and by_row[3]["variance"] == 0.03
    assert by_row[4]["status"] == "MISSING_DOC"
    assert by_row[5]["status"] == "EXCEPTION" and by_row[5]["variance"] == -150.0
    # Fuzzy vendor verified purely because invoice no. + amount corroborate.
    assert by_row[6]["status"] == "VERIFIED", "exact inv#+amount must verify fuzzy vendor"
    # Both postings of INV-777 flagged.
    assert by_row[7]["status"] == "POTENTIAL_DUPLICATE_CLAIM"
    assert by_row[8]["status"] == "POTENTIAL_DUPLICATE_CLAIM"
    assert "rows 7, 8" in by_row[7]["audit_notes"], "duplicate must cite all claimants"
    # Bug #1 regression: the no-invoice line must NOT steal the Uline document...
    assert by_row[9]["status"] == "MISSING_DOC", "weak vendor near-miss must not attach"
    # ...and Uline's real line must stay VERIFIED (not demoted to a duplicate).
    assert by_row[10]["status"] == "VERIFIED", "victim invoice line must stay verified"
    # Bug #4 regression: penny-exact from float noise is VERIFIED, not within-tol.
    assert by_row[11]["status"] == "VERIFIED" and by_row[11]["variance"] == 0.0
    # Key precedence: acronym vendor (~29%) verifies on exact invoice# + amount.
    assert by_row[12]["status"] == "VERIFIED", "key precedence must verify HP acronym"
    assert "vendor name differs" in by_row[12]["audit_notes"], "must log vendor note"

    # The scanned copy is a DUPLICATE_DOCUMENT (informational), NOT a payment
    # claim and NOT unrecorded.
    dup_docs = [r for r in report["results"] if r["status"] == "DUPLICATE_DOCUMENT"]
    assert dup_docs and "beta_777_SCAN.png" in (dup_docs[0]["matching_invoice_file"] or "")
    assert by_row.get(0) is None  # document rows carry no ledger row index

    # The errored invoice lives in its own PROCESSING_ERROR bucket.
    err_rows = [r for r in report["results"] if r["status"] == "PROCESSING_ERROR"]
    assert err_rows and "blurry_scan.pdf" in (err_rows[0]["matching_invoice_file"] or "")
    assert "AI EXTRACTION FAILURE" in err_rows[0]["audit_notes"]

    assert s["tolerance_absolute"] == 0.05 and s["tolerance_relative_pct"] == 0.005

    print("\n[self-test] All assertions passed. Every status path verified. [OK]")
