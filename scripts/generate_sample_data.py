"""
Generate sample test data for the Ledger-to-Invoice Vouching Engine.

Creates, under ``sample_data/``:
  * sample_ledger.xlsx  — a messy General Ledger (SAP/QuickBooks-style headers)
  * invoices/*.pdf      — digital-text invoice PDFs (readable by PyMuPDF, so NO
                          Tesseract install is required to test the happy path)

The data is intentionally crafted so a run surfaces every status:

    Excel row  Vendor        Inv      Ledger $    Invoice $   -> Expected status
    ---------  ------------  -------  ----------  ----------  ----------------------------
    2          Acme Corp     INV-001    1,250.00    1,250.00  VERIFIED
    3          Globex LLC    INV-002      500.03      500.00  VERIFIED_WITHIN_TOLERANCE
    4          Initech       INV-404    3,400.00     (none)   MISSING_DOC
    5          Wayne Ent     INV-003      750.00      900.00  EXCEPTION (amount)
    6          Acme Corp     INV-001    1,250.00    1,250.00  POTENTIAL_DUPLICATE_CLAIM
    (invoice Umbrella INV-777 9,999.99 has no ledger line) -> UNRECORDED_INVOICE

Run:  python scripts/generate_sample_data.py
"""

import os

import fitz  # PyMuPDF
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")
INVOICE_DIR = os.path.join(SAMPLE_DIR, "invoices")


def make_ledger() -> str:
    """Write a messy-but-realistic General Ledger workbook; return its path."""
    df = pd.DataFrame(
        {
            # Deliberately messy headers to exercise the column normalizer.
            "Posting Date": ["2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19"],
            "Supplier Name": ["Acme Corp", "Globex LLC", "Initech", "Wayne Ent", "Acme Corp"],
            "Amount (USD)": [1250.00, 500.03, 3400.00, 750.00, 1250.00],
            "Inv. No.": ["INV-001", "INV-002", "INV-404", "INV-003", "INV-001"],
            # GL Account head. Globex (row 2) is deliberately mis-booked to
            # "IT Equipment" though its invoice is office supplies -> the GL check
            # should flag GL_POSSIBLE_MISMATCH. Wayne is correctly booked to
            # Maintenance Supplies -> GL_CONSISTENT.
            "GL Account": [
                "7200 Software Subscriptions", "6000 IT Equipment",
                "6500 Shipping & Freight", "6300 Maintenance Supplies",
                "7200 Software Subscriptions",
            ],
            "Memo": ["", "", "", "", "duplicate booking?"],  # unmatched column, ignored
        }
    )
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    path = os.path.join(SAMPLE_DIR, "sample_ledger.xlsx")
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def make_invoice_pdf(filename: str, vendor: str, invoice_no: str, date: str,
                     line_items: list[tuple[str, float]], total: float) -> str:
    """Create a simple, digital-text invoice PDF (no scanning/OCR needed)."""
    os.makedirs(INVOICE_DIR, exist_ok=True)
    path = os.path.join(INVOICE_DIR, filename)

    doc = fitz.open()
    page = doc.new_page()  # default A4
    text = [
        vendor.upper(),
        "123 Commerce Street, Business City",
        "",
        "INVOICE",
        f"Invoice #: {invoice_no}",
        f"Date: {date}",
        "Bill To: Client Industries Ltd.",
        "-" * 44,
    ]
    for desc, amount in line_items:
        text.append(f"{desc:<32}{amount:>10,.2f}")
    text.append("-" * 44)
    text.append(f"{'TOTAL DUE':<32}{total:>10,.2f}")

    page.insert_text((72, 90), "\n".join(text), fontsize=11, fontname="courier")
    doc.save(path)
    doc.close()
    return path


def main() -> None:
    ledger_path = make_ledger()
    print(f"[sample] Ledger written:  {ledger_path}")

    invoices = [
        # (file, vendor, inv_no, date, line_items, total)
        ("acme_INV-001.pdf", "Acme Corporation", "INV-001", "15 Jan 2026",
         [("Annual software subscription, 25 seats", 1250.00)], 1250.00),
        # Office supplies — but the ledger books this to "IT Equipment" -> mismatch.
        ("globex_INV-002.pdf", "Globex LLC", "INV-002", "16 Jan 2026",
         [("Copy paper cases", 300.00), ("Toner cartridges and pens", 200.00)], 500.00),
        # Maintenance content matching its booked head -> consistent.
        ("wayne_INV-003.pdf", "Wayne Enterprises", "INV-003", "17 Jan 2026",
         [("Facility maintenance and repair parts", 900.00)], 900.00),
        ("umbrella_INV-777.pdf", "Umbrella Inc", "INV-777", "20 Jan 2026",
         [("Lab equipment", 9999.99)], 9999.99),
    ]
    for args in invoices:
        p = make_invoice_pdf(*args)
        print(f"[sample] Invoice written: {p}")

    print("\n[sample] Done. Upload sample_data/sample_ledger.xlsx as the ledger")
    print("[sample] and ALL files in sample_data/invoices/ as the invoices.")


if __name__ == "__main__":
    main()
