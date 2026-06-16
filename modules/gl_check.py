"""
Audit Agent: Ledger-to-Invoice Vouching Engine
Module 5 — GL Account Consistency Check (advisory)
==================================================

This module compares the General Ledger account head a transaction was *booked*
to against the expense category *inferred* from the matched invoice's content
(vendor + line-item descriptions), and flags likely miscodings for human review.

This is JUDGMENT, not a hard match. Account classification is inherently fuzzy
and auditors disagree on edge cases, so the output is deliberately ADVISORY and
uses three states — never a pass/fail:

    GL_CONSISTENT        invoice content aligns with the booked head
    GL_POSSIBLE_MISMATCH content confidently suggests a different head than booked
    GL_UNDETERMINED      not enough signal to judge (we do NOT force a guess)

It NEVER auto-corrects the ledger — it only flags for a human. The bias is
intentionally toward GL_UNDETERMINED: a false "miscoding" alarm is worse than
silence here.

The category vocabulary and the keyword/vendor → category maps below are a
clearly-separated, EDITABLE config so a real chart of accounts can be tuned (or
swapped in) without touching the logic.

Public entry point:
    check_gl_account(booked_head, vendor_name, line_items) -> dict
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
GL_CONSISTENT = "GL_CONSISTENT"
GL_POSSIBLE_MISMATCH = "GL_POSSIBLE_MISMATCH"
GL_UNDETERMINED = "GL_UNDETERMINED"

# ===========================================================================
# EDITABLE CONFIG — chart-of-accounts category vocabulary and signals.
# Swap/extend these to match a real client chart of accounts. The keys of
# CATEGORY_KEYWORDS are the canonical account heads the checker understands; a
# booked head outside this vocabulary yields GL_UNDETERMINED (we don't guess
# against an unknown chart).
# ===========================================================================
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "IT Equipment": [
        "server", "laptop", "desktop", "workstation", "switch", "router",
        "monitor", "hardware", "computer", "ssd", "gpu", "docking", "keyboard",
    ],
    "Cloud Hosting": [
        "ec2", "compute", "s3", "storage", "egress", "bigquery", "cloud",
        "hosting", "instance", "kubernetes", "lambda", "azure", "gcp",
        "bandwidth", "vm",
    ],
    "Shipping & Freight": [
        "freight", "ground", "overnight", "liftgate", "ltl", "shipping",
        "courier", "parcel", "delivery", "express", "pallet",
    ],
    "Office Supplies": [
        "paper", "pens", "toner", "ink", "breakroom", "stapler", "notebook",
        "binder", "cartridge", "envelope", "office supplies",
    ],
    "Maintenance Supplies": [
        "maintenance", "repair", "lubricant", "fastener", "bolt", "filter",
        "hvac", "janitorial", "cleaning", "grease", "valve",
    ],
    "Software Subscriptions": [
        "subscription", "annual plan", "seats", "users", "license", "saas",
        "software", "renewal", "esignature", "plan",
    ],
    "Rent & Facilities": [
        "rent", "lease", "facility", "workspace", "coworking", "office space",
        "desk membership",
    ],
    "Utilities & Telecom": [
        "internet", "phone", "telecom", "utility", "electricity", "broadband",
        "wireless", "data plan", "fiber",
    ],
    "Records & Storage": [
        "shredding", "records storage", "bin", "archive", "document storage",
        "records management", "destruction",
    ],
    "Fuel & Vehicle": [
        "fuel", "gasoline", "diesel", "vehicle", "fleet", "mileage", "gallon",
    ],
    "Marketing & Recruiting": [
        "marketing", "advertising", "recruiting", "job posting", "campaign",
        "ads", "sponsored",
    ],
}

# Vendor substring -> category. A vendor signal counts as one corroborating hit.
VENDOR_CATEGORY: dict[str, str] = {
    "amazon web services": "Cloud Hosting", "aws": "Cloud Hosting",
    "google cloud": "Cloud Hosting", "azure": "Cloud Hosting",
    "dell": "IT Equipment", "hewlett": "IT Equipment", "hp inc": "IT Equipment",
    "cdw": "IT Equipment", "lenovo": "IT Equipment",
    "fedex": "Shipping & Freight", "ups": "Shipping & Freight",
    "dhl": "Shipping & Freight",
    "staples": "Office Supplies", "office depot": "Office Supplies",
    "quill": "Office Supplies",
    "grainger": "Maintenance Supplies", "fastenal": "Maintenance Supplies",
    "uline": "Shipping & Freight",
    "adobe": "Software Subscriptions", "docusign": "Software Subscriptions",
    "slack": "Software Subscriptions", "microsoft": "Software Subscriptions",
    "zoom": "Software Subscriptions",
    "wework": "Rent & Facilities",
    "comcast": "Utilities & Telecom", "verizon": "Utilities & Telecom",
    "at&t": "Utilities & Telecom",
    "iron mountain": "Records & Storage",
    "shell": "Fuel & Vehicle", "chevron": "Fuel & Vehicle", "bp": "Fuel & Vehicle",
    "linkedin": "Marketing & Recruiting", "indeed": "Marketing & Recruiting",
}

# Canonical lookup for booked-head normalization (lowercased -> canonical name).
_HEAD_BY_LOWER = {name.lower(): name for name in CATEGORY_KEYWORDS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_head(booked_head) -> str:
    """
    Strip a GL code prefix and surrounding noise from a booked account head.

        "6300 Maintenance Supplies" -> "Maintenance Supplies"
        "6300 - Maintenance Supplies" -> "Maintenance Supplies"
        "Cloud Hosting" -> "Cloud Hosting"

    Returns the cleaned text (empty string if blank/None).
    """
    if booked_head is None:
        return ""
    text = str(booked_head).strip()
    # Remove a leading numeric code and any separator (space, dash, colon).
    text = re.sub(r"^\s*\d+\s*[-:.]?\s*", "", text)
    return text.strip()


def infer_category(
    vendor_name: Optional[str], line_items: Optional[str]
) -> tuple[Optional[str], str, list[str]]:
    """
    Infer the expense category from invoice content (line items + vendor).

    Returns (category, confidence, signals):
        category    canonical head, or None when there is no signal at all
        confidence  "strong" (>=2 corroborating signals), "weak" (exactly 1),
                    or "none"
        signals     the matched keywords/vendor tokens that drove the inference
    """
    text_l = (line_items or "").lower()
    vendor_l = (vendor_name or "").lower()

    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    # Content keyword signals.
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = [kw for kw in kws if kw in text_l]
        if hits:
            scores[cat] = scores.get(cat, 0) + len(hits)
            matched.setdefault(cat, []).extend(hits)

    # Vendor signal (counts as one corroborating hit).
    for key, cat in VENDOR_CATEGORY.items():
        if key in vendor_l:
            scores[cat] = scores.get(cat, 0) + 1
            matched.setdefault(cat, []).append(f"vendor~{key}")
            break

    if not scores:
        return None, "none", []

    best = max(scores, key=lambda c: scores[c])
    best_score = scores[best]
    confidence = "strong" if best_score >= 2 else "weak"
    return best, confidence, matched.get(best, [])


def _result(status: str, booked: str, suggested: str, notes: str) -> dict:
    return {
        "gl_status": status,
        "gl_booked": booked,
        "gl_suggested": suggested,
        "gl_notes": notes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_gl_account(
    booked_head, vendor_name: Optional[str], line_items: Optional[str]
) -> dict:
    """
    Advisory GL account consistency check for a single matched ledger line.

    Args:
        booked_head: The GL account head as booked on the ledger (may carry a
                     numeric code prefix, e.g. "6300 Maintenance Supplies").
        vendor_name: The matched invoice's vendor (seller).
        line_items:  Plain-text summary of the invoice's line items.

    Returns:
        {"gl_status", "gl_booked", "gl_suggested", "gl_notes"} where gl_status is
        one of GL_CONSISTENT / GL_POSSIBLE_MISMATCH / GL_UNDETERMINED.

    Decision rules (biased toward UNDETERMINED on thin evidence):
        - no booked head, or booked head outside the known chart -> UNDETERMINED
        - no inferred category -> UNDETERMINED
        - inferred == booked -> CONSISTENT
        - inferred != booked and confidence strong -> POSSIBLE_MISMATCH
        - inferred != booked and confidence weak -> UNDETERMINED (don't cry wolf)
    """
    booked = normalize_head(booked_head)
    if not booked:
        return _result(GL_UNDETERMINED, "", "", "No GL account booked on the ledger line.")

    canonical_booked = _HEAD_BY_LOWER.get(booked.lower())
    if canonical_booked is None:
        return _result(
            GL_UNDETERMINED, booked, "",
            f"Booked head '{booked}' is not in the configured chart of accounts; "
            "cannot assess consistency.",
        )

    inferred, confidence, signals = infer_category(vendor_name, line_items)
    if inferred is None:
        return _result(
            GL_UNDETERMINED, canonical_booked, "",
            "Insufficient invoice content to infer an expense category.",
        )

    signal_str = ", ".join(signals[:5])

    if inferred == canonical_booked:
        return _result(
            GL_CONSISTENT, canonical_booked, "",
            f"Invoice content aligns with the booked head (signals: {signal_str}).",
        )

    if confidence == "strong":
        return _result(
            GL_POSSIBLE_MISMATCH, canonical_booked, inferred,
            f"Booked to '{canonical_booked}', but invoice content suggests "
            f"'{inferred}' (signals: {signal_str}). Possible miscoding — review.",
        )

    # Different but only weak evidence -> stay quiet.
    return _result(
        GL_UNDETERMINED, canonical_booked, "",
        f"Weak signal hints at '{inferred}' vs booked '{canonical_booked}', but "
        "evidence is insufficient to flag a mismatch.",
    )


# ---------------------------------------------------------------------------
# Isolated self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Clearly correct: AWS cloud line items booked to Cloud Hosting -> CONSISTENT
    r = check_gl_account(
        "7100 Cloud Hosting", "Amazon Web Services",
        "EC2 compute, S3 storage, data egress",
    )
    print("AWS/Cloud:", r["gl_status"], "|", r["gl_notes"])
    assert r["gl_status"] == GL_CONSISTENT

    # 2) Clear miscode: office paper/toner booked to IT Equipment -> POSSIBLE_MISMATCH
    r = check_gl_account(
        "6000 IT Equipment", "Quill LLC",
        "Copy paper cases, toner cartridges, pens",
    )
    print("Office->IT:", r["gl_status"], "| suggested:", r["gl_suggested"])
    assert r["gl_status"] == GL_POSSIBLE_MISMATCH
    assert r["gl_suggested"] == "Office Supplies"

    # 3) Genuinely ambiguous: no usable content, generic vendor -> UNDETERMINED
    r = check_gl_account("6000 IT Equipment", "Acme Holdings", "miscellaneous services")
    print("Ambiguous:", r["gl_status"], "|", r["gl_notes"])
    assert r["gl_status"] == GL_UNDETERMINED

    # 4) Booked head outside the configured chart -> UNDETERMINED (no false alarm)
    r = check_gl_account("9999 Intercompany Settlements", "Dell", "laptop, monitor")
    assert r["gl_status"] == GL_UNDETERMINED

    # 5) Code-prefix normalization
    assert normalize_head("6300 Maintenance Supplies") == "Maintenance Supplies"
    assert normalize_head("6300 - Maintenance Supplies") == "Maintenance Supplies"

    # 6) Weak-but-different stays UNDETERMINED (single vendor hit, conflicting book)
    r = check_gl_account("6000 IT Equipment", "Uline", "")
    print("Weak/diff:", r["gl_status"])
    assert r["gl_status"] == GL_UNDETERMINED, "single weak signal must not flag mismatch"

    print("\n[self-test] GL check: all assertions passed. [OK]")
