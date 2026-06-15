"""
Audit Agent: Three-Way Matcher — core processing modules.

This package bundles the four pipeline stages:
    1. ledger_parser  — ingest & normalize the client General Ledger (.xlsx)
    2. document_ocr   — OCR invoices & extract structured metadata via Groq
    3. match_engine   — deterministic three-way reconciliation
    4. report_gen     — render the styled Excel audit workpaper
"""

from . import ledger_parser, document_ocr, match_engine, report_gen

__all__ = ["ledger_parser", "document_ocr", "match_engine", "report_gen"]
