"""
Command-line test harness for the Ledger-to-Invoice Vouching Engine.

Runs the full backend pipeline (parse -> extract -> reconcile -> report) against
REAL files on disk, printing what was parsed and what the AI extracted from each
invoice so you can diagnose real-world data quality WITHOUT the Streamlit UI.

Usage:
    python scripts/run_cli.py --ledger PATH.xlsx --invoices FOLDER_OR_FILES...

Examples:
    python scripts/run_cli.py --ledger data/GL_Q1.xlsx --invoices data/invoices
    python scripts/run_cli.py --ledger gl.xlsx --invoices a.pdf b.pdf --sheet "Sheet1"
    python scripts/run_cli.py --ledger gl.xlsx --invoices inv/ --abs-tol 0.05 --rel-tol 0.5

The Groq API key is read from .env (GROQ_API_KEY), exactly like the app.
"""

import argparse
import glob
import os
import sys

# Make the project root importable when run as `python scripts/run_cli.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import ledger_parser, document_ocr, match_engine, report_gen  # noqa: E402

INVOICE_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")


def _collect_invoices(paths: list[str]) -> list[str]:
    """Expand any folders in ``paths`` into the invoice files they contain."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in INVOICE_EXTS:
                files.extend(glob.glob(os.path.join(p, f"*{ext}")))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"  ! skipping (not found): {p}")
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser(description="Vouching Engine CLI test harness")
    ap.add_argument("--ledger", required=True, help="Path to the General Ledger .xlsx")
    ap.add_argument("--invoices", required=True, nargs="+",
                    help="Invoice files and/or folders containing them")
    ap.add_argument("--sheet", default=None, help="Ledger worksheet name (optional)")
    ap.add_argument("--abs-tol", type=float, default=0.05, help="Absolute $ tolerance")
    ap.add_argument("--rel-tol", type=float, default=0.5,
                    help="Relative tolerance as a PERCENT (e.g. 0.5 = 0.5%%)")
    ap.add_argument("--out", default="cli_audit_workpaper.xlsx", help="Output .xlsx path")
    args = ap.parse_args()

    # --- API key pre-flight --------------------------------------------------
    if not document_ocr.get_groq_api_key():
        print("ERROR: No Groq API key. Set GROQ_API_KEY in your .env file.")
        return 2

    # --- Step 1: parse the ledger -------------------------------------------
    print(f"\n[1/4] Parsing ledger: {args.ledger}")
    try:
        ledger = ledger_parser.parse_general_ledger(args.ledger, target_sheet=args.sheet)
    except Exception as exc:  # noqa: BLE001
        print(f"  LEDGER PARSE FAILED: {exc}")
        return 1
    print(f"  -> {len(ledger)} ledger rows parsed.")
    if ledger:
        print(f"  -> sample row: {ledger[0]}")

    # --- Step 2: extract invoices (LIVE Groq calls) -------------------------
    invoice_files = _collect_invoices(args.invoices)
    print(f"\n[2/4] Extracting {len(invoice_files)} invoice(s) via Groq "
          f"({document_ocr.GROQ_MODEL}):")
    invoices = []
    for path in invoice_files:
        name = os.path.basename(path)
        try:
            meta = document_ocr.extract_invoice_data(path)
        except Exception as exc:  # noqa: BLE001
            meta = {"vendor_name": None, "invoice_number": None, "date": None,
                    "total_amount": None, "processing_error": True,
                    "error_detail": f"{type(exc).__name__}: {exc}"}
        meta["source_file"] = name
        invoices.append(meta)
        flag = "  ⚠ ERROR" if meta.get("processing_error") else ""
        print(f"  - {name:32} vendor={meta.get('vendor_name')!r} "
              f"inv={meta.get('invoice_number')!r} total={meta.get('total_amount')!r}{flag}")
        if meta.get("processing_error"):
            print(f"      detail: {meta.get('error_detail')}")

    # --- Step 3: reconcile ---------------------------------------------------
    print(f"\n[3/4] Reconciling (abs ${args.abs_tol:.2f} / rel {args.rel_tol:.3g}%)…")
    rec = match_engine.reconcile_ledger_with_invoices(
        ledger, invoices,
        absolute_tolerance=args.abs_tol,
        relative_tolerance_pct=args.rel_tol / 100.0,
    )
    s = rec["summary"]
    print(f"  verified={s['verified_count']}  within_tol={s['verified_within_tolerance_count']}  "
          f"exceptions={s['exception_count']}  missing={s['missing_doc_count']}")
    print(f"  duplicates={s['potential_duplicate_count']}  unrecorded={s['unrecorded_invoice_count']}  "
          f"errors={s['processing_error_count']}")
    print("\n  Per-line results:")
    for r in rec["results"]:
        row = r["ledger_row_index"] if r["ledger_row_index"] is not None else "—"
        print(f"   [{r['status']:26}] row={row} {r['ledger_vendor']!r} "
              f"${r['ledger_amount']:,.2f} var={r['variance']:,.2f}")

    # --- Step 4: write the workpaper ----------------------------------------
    out_path = report_gen.generate_audit_report(rec, args.out)
    print(f"\n[4/4] Workpaper written: {out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
