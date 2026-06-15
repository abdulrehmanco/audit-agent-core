# 🧾 Audit Agent: Three-Way Matcher

An AI-assisted **substantive testing** tool for corporate auditors. It reconciles a
client's **General Ledger** against their **source invoices** and produces a clean,
color-coded Excel **audit workpaper** — turning a tedious manual tie-out into a
few-second automated run.

> **Determinism where it counts.** AI/OCR is used only to *read* messy documents and
> *find* candidate matches. Every financial verdict (does the amount tie out to the
> penny?) is rendered by pure, auditable arithmetic — no probabilistic guessing on money.

---

## Architecture

A four-stage pipeline behind a Streamlit web UI:

| # | Module | Responsibility |
|---|--------|----------------|
| 1 | `modules/ledger_parser.py` | Ingest the GL `.xlsx`, normalize messy column names (SAP/Oracle/QuickBooks variants), coerce to safe types, preserve the physical Excel row number. |
| 2 | `modules/document_ocr.py` | Extract text from invoices — PyMuPDF for digital PDFs, Tesseract OCR fallback for scans/images — then structure it via Groq `llama-3.3-70b-versatile` into strict JSON. |
| 3 | `modules/match_engine.py` | Deterministically reconcile each ledger line → `VERIFIED` / `EXCEPTION` / `MISSING_DOC`. Fuzzy matching (`thefuzz`) only to *locate* candidates; math is exact. |
| 4 | `modules/report_gen.py` | Render a professionally styled Excel workpaper (navy headers, currency formats, conditional color-coding, auto-fit columns). |
| — | `app.py` | Streamlit front end: upload, configure, run, view metrics, download. |

```
audit-agent-core/
├── modules/
│   ├── __init__.py
│   ├── ledger_parser.py
│   ├── document_ocr.py
│   ├── match_engine.py
│   └── report_gen.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Prerequisites
- **Python 3.10+** (tested on 3.13)
- **Tesseract OCR binary** — required only for scanned/image invoices (not a pip package):
  - Windows: <https://github.com/UB-Mannheim/tesseract/wiki>
  - macOS: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`
  - Confirm it's on your PATH: `tesseract --version`
- **Groq API key** — get one at <https://console.groq.com>

### 2. Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (default <http://localhost:8501>).

---

## Usage

1. Paste your **Groq API key** into the sidebar.
2. (Optional) Set the worksheet name and matching thresholds.
3. Upload the **General Ledger** (`.xlsx`) and one or more **invoice** files (PDF/PNG/JPG…).
4. Click **🚀 Run Automated Three-Way Match**.
5. Review the metrics dashboard and **download the Excel workpaper**.

### Expected ledger columns
The parser auto-maps common header variations. At minimum it needs an **amount** and a
**vendor** column. Recognized aliases include:

| Canonical | Accepted variations |
|-----------|--------------------|
| `date` | date, transaction_date, tx_date, posting_date |
| `vendor` | vendor, supplier, payee, vendor_name, description |
| `amount` | amount, amount_usd, total, debit, value |
| `invoice_no` | invoice_no, invoice_number, inv_no, reference, ref_no |

---

## Data privacy

**Zero-retention by design.** Uploaded ledgers and invoices are written to a per-run
temporary directory only for the moment the parsing libraries need a real file path, then
the entire directory is deleted in a `finally`/`with` teardown the instant the run ends.
Nothing is written to a database. The only client data that leaves the machine is the
invoice **text** sent to the Groq API for extraction (see the architecture review for the
data-residency implications of this).

---

## Testing

Each module ships a self-contained self-test. Run any of them directly:

```bash
python modules/ledger_parser.py     # mock GL → asserts parsing, row indices, KeyError
python modules/match_engine.py      # mock data → asserts VERIFIED / EXCEPTION / MISSING_DOC
python modules/report_gen.py        # writes test_audit_workpaper.xlsx and re-verifies styling
python modules/document_ocr.py      # offline JSON-coercion test (set GROQ_API_KEY for live)
```

---

## Status & roadmap

This is a working **MVP / proof-of-concept**. Before production use at an audit firm, see
[`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) for the gaps that must be closed
(auditability, data residency, idempotent matching, verification of AI output, etc.).
