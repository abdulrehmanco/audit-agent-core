"""
Audit Agent: Automated Ledger-to-Invoice Vouching Engine — Streamlit Web Interface
==================================================================================

This is the public-facing web application that ties the four core modules into
a single, auditor-friendly workflow:

    1. modules.ledger_parser.parse_general_ledger        (ingest the GL .xlsx)
    2. modules.document_ocr.extract_invoice_data         (OCR + Groq extraction)
    3. modules.match_engine.reconcile_ledger_with_invoices  (deterministic match)
    4. modules.report_gen.generate_audit_report          (styled Excel workpaper)

Data-privacy posture (ZERO-RETENTION)
-------------------------------------
No client ledger, invoice image, or generated workpaper is ever persisted to a
database or left on disk. Uploaded files are written to a per-run temporary
directory only for the brief moment the parsing libraries need a real path, and
that entire directory is deleted in a ``finally`` block the instant processing
finishes (success or failure). The only thing that survives a run is the report
held in memory for the download button, which is released when the session ends.

API protection
--------------
A short mandatory pause is inserted between invoice API calls to stay under
Groq's free-tier rate limits and avoid HTTP 429 storms on large batches.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback

import pandas as pd
import streamlit as st

# Load .env into the environment before anything reads the API key. (The OCR
# module also calls this on import; doing it here too keeps app.py self-contained.)
try:
    from dotenv import load_dotenv

    # override=True so edits to .env take effect on the next rerun without having
    # to restart the whole Streamlit process.
    load_dotenv(override=True)
except ImportError:
    pass


def _bridge_streamlit_secrets() -> None:
    """
    Copy Streamlit Cloud secrets into the environment for the backend modules.

    Locally the config comes from .env; on Streamlit Community Cloud there is no
    .env (it's git-ignored), so the key is set in the app's *Secrets* UI and read
    via ``st.secrets``. The backend only reads ``os.environ``, so we bridge the
    two here. Safe no-op when no secrets are configured.
    """
    try:
        for key in ("GROQ_API_KEY", "GROQ_MODEL", "TESSERACT_CMD"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no secrets file locally is fine
        pass


_bridge_streamlit_secrets()

# --- Local module imports (standard package-relative paths) ----------------
from modules import ledger_parser, document_ocr, match_engine, report_gen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "Automated Ledger-to-Invoice Vouching Engine"
SUPPORTED_INVOICE_TYPES = ["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]
# Mandatory courtesy pause (seconds) between Groq calls to respect rate limits.
RATE_LIMIT_PAUSE_SECONDS = 2
REPORT_FILENAME = "Audit_Workpaper.xlsx"

# Soft pastel backgrounds per status (mirrors the Excel workpaper color-coding).
STATUS_COLORS = {
    "VERIFIED": "#E2EFDA",
    "VERIFIED_WITHIN_TOLERANCE": "#D5E8D4",
    "EXCEPTION": "#FFF2CC",
    "MISSING_DOC": "#FCE4D6",
    "POTENTIAL_DUPLICATE_CLAIM": "#FAC090",
    "DUPLICATE_DOCUMENT": "#DDEBF7",
    "UNRECORDED_INVOICE": "#E4DFEC",
    "PROCESSING_ERROR": "#E7E6E6",
    "NEEDS_REVIEW": "#E7E6E6",
}

# Display order + friendly headers for the on-screen reconciliation table.
RESULT_COLUMNS = [
    ("ledger_row_index", "Excel Row"),
    ("ledger_vendor", "Vendor"),
    ("ledger_amount", "Amount"),
    ("ledger_invoice_no", "Invoice #"),
    ("status", "Status"),
    ("variance", "Variance"),
    ("matching_invoice_file", "Matching File"),
    ("audit_notes", "Audit Notes"),
]


# ---------------------------------------------------------------------------
# Page configuration & light theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default chrome (top-right toolbar incl. the GitHub "Fork"
# button and hamburger menu, plus the bottom "Made with Streamlit" badge/footer)
# for a clean, white-labelled look. We deliberately do NOT hide the whole header
# so the mobile sidebar-open control still works.
_HIDE_STREAMLIT_CHROME = """
    <style>
      [data-testid="stToolbar"] {display: none !important;}
      #MainMenu {visibility: hidden !important;}
      [data-testid="stStatusWidget"] {display: none !important;}
      [data-testid="stDecoration"] {display: none !important;}
      footer {visibility: hidden !important; height: 0 !important;}
      /* "Hosted with / Made with Streamlit" viewer badge (class names vary by
         version, so match broadly) */
      [class*="viewerBadge"] {display: none !important;}
      a[href*="streamlit.io/cloud"],
      a[href*="share.streamlit.io"] {display: none !important;}
    </style>
"""
st.markdown(_HIDE_STREAMLIT_CHROME, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: persist a Streamlit UploadedFile to a temp dir and return its path.
# ---------------------------------------------------------------------------
def _write_temp(upload, directory: str) -> str:
    """
    Write an uploaded file's bytes into ``directory`` and return the full path.

    The parsing/OCR libraries (openpyxl, PyMuPDF, Tesseract) operate on real
    file paths, so we briefly materialize the upload. The containing directory
    is deleted by the caller as soon as the run completes (zero-retention).

    Args:
        upload:    A Streamlit UploadedFile object.
        directory: The temporary directory to write into.

    Returns:
        The absolute path to the written temp file.
    """
    safe_name = os.path.basename(upload.name)
    path = os.path.join(directory, safe_name)
    with open(path, "wb") as fh:
        fh.write(upload.getbuffer())
    return path


# ---------------------------------------------------------------------------
# The unified processing pipeline (Steps 1-4).
# ---------------------------------------------------------------------------
def run_pipeline(
    ledger_upload,
    invoice_uploads,
    target_sheet: str | None,
    vendor_threshold: int,
    absolute_tolerance: float,
    relative_tolerance_pct: float,
    status_placeholder,
) -> tuple[dict, bytes]:
    """
    Execute the full vouching run and return (reconciliation, report_bytes).

    All temporary files live inside a single ``TemporaryDirectory`` that is
    torn down automatically on exit — nothing client-related touches permanent
    storage. The Groq API key is read from .env inside the OCR module; it is
    never passed in from the UI.

    Args:
        ledger_upload:          UploadedFile for the General Ledger .xlsx.
        invoice_uploads:        List of UploadedFile invoice documents.
        target_sheet:           Optional worksheet name in the ledger workbook.
        vendor_threshold:       Vendor-similarity % required for a clean match.
        absolute_tolerance:     Allowed absolute $ variance (e.g. 0.05).
        relative_tolerance_pct: Allowed variance as a fraction (e.g. 0.005 = 0.5%).
        status_placeholder:     A Streamlit container used for live status text.

    Returns:
        (reconciliation_data, report_bytes)
    """
    # Apply the user's vendor-match strictness to the deterministic engine.
    match_engine.VENDOR_MATCH_THRESHOLD = vendor_threshold

    with tempfile.TemporaryDirectory(prefix="audit_agent_run_") as work_dir:
        # --- Step 1: Parse the General Ledger --------------------------------
        status_placeholder.info("📒 Step 1/4 — Parsing the General Ledger…")
        ledger_path = _write_temp(ledger_upload, work_dir)
        ledger_data = ledger_parser.parse_general_ledger(
            ledger_path, target_sheet=target_sheet or None
        )

        # --- Step 2: OCR + extract every invoice -----------------------------
        invoice_data: list[dict] = []
        total = len(invoice_uploads)
        progress = st.progress(0.0, text="Preparing to read invoices…")

        for idx, upload in enumerate(invoice_uploads, start=1):
            status_placeholder.info(
                f"📄 Step 2/4 — Extracting invoice {idx} of {total}: "
                f"**{upload.name}**"
            )
            invoice_path = _write_temp(upload, work_dir)
            try:
                # API key is resolved from .env inside the OCR module.
                metadata = document_ocr.extract_invoice_data(invoice_path)
            except Exception as exc:  # noqa: BLE001 - never let one file kill the batch
                # A hard failure (unreadable file, no text, no key) is a TOOLING
                # error — flag it so the engine reports PROCESSING_ERROR, never a
                # false MISSING_DOC.
                st.warning(f"⚠️ Could not extract '{upload.name}': {exc}")
                metadata = {
                    "vendor_name": None,
                    "invoice_number": None,
                    "date": None,
                    "total_amount": None,
                    "processing_error": True,
                    "error_detail": f"{type(exc).__name__}: {exc}",
                }

            # Surface a soft warning for extraction failures the OCR module
            # flagged internally (e.g. a caught 429) so the auditor sees them.
            if metadata.get("processing_error"):
                st.warning(
                    f"⚠️ '{upload.name}' flagged for review — "
                    f"{metadata.get('error_detail', 'AI extraction failure')}"
                )

            # Tag with the ORIGINAL filename so the report can cite the source
            # document (match_engine reads the 'source_file' key).
            metadata["source_file"] = upload.name
            invoice_data.append(metadata)

            progress.progress(idx / total, text=f"Processed {idx}/{total} invoices")

            # API protection: brief mandatory pause between calls (skip after
            # the final file, no need to wait for nothing).
            if idx < total:
                time.sleep(RATE_LIMIT_PAUSE_SECONDS)

        progress.empty()

        # --- Step 3: Deterministic reconciliation ----------------------------
        status_placeholder.info("⚖️ Step 3/4 — Reconciling ledger against invoices…")
        reconciliation = match_engine.reconcile_ledger_with_invoices(
            ledger_data,
            invoice_data,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance_pct=relative_tolerance_pct,
        )

        # --- Step 4: Generate the styled Excel workpaper ---------------------
        status_placeholder.info("📊 Step 4/4 — Building the audit workpaper…")
        report_path = os.path.join(work_dir, REPORT_FILENAME)
        report_gen.generate_audit_report(reconciliation, report_path)

        # Read the report bytes into memory BEFORE the temp dir is destroyed.
        with open(report_path, "rb") as fh:
            report_bytes = fh.read()

    # <- TemporaryDirectory (and every client file inside it) is now deleted.
    return reconciliation, report_bytes


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    """Render the configuration sidebar and return the collected settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Ledger Options")
        target_sheet = st.text_input(
            "Worksheet name (optional)",
            value="",
            help="Leave blank to use the first sheet in the workbook.",
        )

        st.subheader("Matching Thresholds")
        vendor_threshold = st.slider(
            "Vendor name match strictness (%)",
            min_value=70,
            max_value=100,
            value=match_engine.VENDOR_MATCH_THRESHOLD,
            help="Minimum fuzzy similarity between the ledger vendor and the "
            "invoice vendor required to count as a clean match. Higher is "
            "stricter.",
        )

        st.subheader("Financial Tolerance")
        st.caption(
            "A line whose amount variance falls within **either** band is marked "
            "**Verified Within Tolerance** (the exact variance is still logged) "
            "rather than an exception — covering rounding, sales tax, and FX. "
            "Set both to 0 for strict to-the-penny matching."
        )
        absolute_tolerance = st.number_input(
            "Absolute tolerance ($)",
            min_value=0.0,
            max_value=1000.0,
            value=float(match_engine.DEFAULT_ABSOLUTE_TOLERANCE),
            step=0.01,
            format="%.2f",
            help="Maximum absolute dollar variance treated as a tolerated match.",
        )
        relative_tolerance_pct_display = st.number_input(
            "Relative tolerance (%)",
            min_value=0.0,
            max_value=25.0,
            value=float(match_engine.DEFAULT_RELATIVE_TOLERANCE_PCT * 100),
            step=0.1,
            format="%.2f",
            help="Maximum variance as a percentage of the invoice amount "
            "(e.g. 0.5% covers small local tax differences on large invoices).",
        )

        st.divider()
        with st.expander("🩺 System diagnostics"):
            tess = document_ocr.get_tesseract_version()
            if tess:
                st.success(f"Tesseract OCR available (v{tess}) — scans will be read.")
            else:
                st.warning(
                    "Tesseract OCR not detected — scanned/photo invoices can't be "
                    "read (digital PDFs still work). On Streamlit Cloud, ensure "
                    "`packages.txt` contains `tesseract-ocr` and reboot the app."
                )
        st.caption(
            "🔒 **Zero-retention:** uploaded files are processed transiently and "
            "deleted immediately after the run. Nothing is saved to a database."
        )

    return {
        "target_sheet": target_sheet.strip(),
        "vendor_threshold": int(vendor_threshold),
        "absolute_tolerance": float(absolute_tolerance),
        # Convert the percentage the auditor typed back into a fraction.
        "relative_tolerance_pct": float(relative_tolerance_pct_display) / 100.0,
    }


# ---------------------------------------------------------------------------
# Results table rendering
# ---------------------------------------------------------------------------
def _render_results_table(results: list) -> None:
    """
    Render the reconciliation results as a free-flowing, status-colored table.

    Uses a wrapping HTML table (rather than ``st.dataframe``) so long fields —
    especially the ``audit_notes`` column — are never truncated. Currency columns
    are formatted, missing ledger row numbers show as an em dash, and each row is
    tinted by its audit status to mirror the Excel workpaper.
    """
    df = pd.DataFrame(results)

    # Order + rename to the friendly headers (only columns that exist).
    ordered = [(k, label) for k, label in RESULT_COLUMNS if k in df.columns]
    df = df[[k for k, _ in ordered]].rename(columns=dict(ordered))

    # Human-friendly value formatting.
    if "Excel Row" in df:
        df["Excel Row"] = df["Excel Row"].apply(
            lambda v: "—" if v is None or pd.isna(v) else int(v)
        )
    for money_col in ("Amount", "Variance"):
        if money_col in df:
            df[money_col] = df[money_col].apply(
                lambda v: f"${v:,.2f}" if isinstance(v, (int, float)) else v
            )
    if "Matching File" in df:
        df["Matching File"] = df["Matching File"].fillna("—")

    # Tint each row by its status (looked up before we restyle the cells).
    statuses = [str(r.get("status", "")).upper() for r in results]

    def _row_style(row):
        color = STATUS_COLORS.get(statuses[row.name], "#FFFFFF")
        return [f"background-color: {color}"] * len(row)

    styler = (
        df.style
        .apply(_row_style, axis=1)
        .hide(axis="index")
        .set_table_attributes('class="audit-results"')
    )
    html_table = styler.to_html()

    # CSS: borders, navy header, and — critically — wrapping cells so nothing is
    # cut off. The Audit Notes column is given room while still wrapping.
    css = """
    <style>
      .audit-results { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
      .audit-results th {
          background-color: #1F497D; color: #FFFFFF; font-weight: 600;
          text-align: left; padding: 8px 10px; border: 1px solid #BFBFBF;
          position: sticky; top: 0;
      }
      .audit-results td {
          padding: 7px 10px; border: 1px solid #D9D9D9;
          white-space: normal; word-wrap: break-word; vertical-align: top;
          color: #1a1a1a;
      }
      .audit-results td:nth-child(8) { min-width: 320px; }  /* Audit Notes */
    </style>
    """
    # A scroll container keeps very long runs manageable without truncating text.
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(
        f'<div style="max-height: 640px; overflow-y: auto;">{html_table}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------
def render_dashboard(reconciliation: dict, report_bytes: bytes) -> None:
    """Render the high-level metrics, download button, and results table."""
    summary = reconciliation.get("summary", {})
    results = reconciliation.get("results", [])

    st.success("✅ Vouching run complete.")

    # --- Metric cards (two rows: existence then completeness/tooling) --------
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Ledger Records", summary.get("total_ledger_records", 0))
    r1c2.metric("✅ Verified", summary.get("verified_count", 0))
    r1c3.metric(
        "🟢 Within Tolerance", summary.get("verified_within_tolerance_count", 0)
    )
    r1c4.metric("⚠️ Exceptions", summary.get("exception_count", 0))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("❌ Missing Docs", summary.get("missing_doc_count", 0))
    r2c2.metric(
        "🟠 Duplicate Claims",
        summary.get("potential_duplicate_count", 0),
        help="Same invoice booked on multiple ledger lines — a payment risk.",
    )
    r2c3.metric("🟣 Unrecorded Invoices", summary.get("unrecorded_invoice_count", 0))
    r2c4.metric("⚙️ Processing Errors", summary.get("processing_error_count", 0))

    # Document copies (e.g. a scan of an already-booked invoice) are informational
    # only — shown separately so they don't inflate the payment-risk count.
    dup_docs = summary.get("duplicate_document_count", 0)
    if dup_docs:
        st.caption(
            f"🔵 {dup_docs} duplicate document(s) detected (e.g. a scan of an "
            "already-recorded invoice) — informational only, not a payment risk."
        )

    # Surface the tolerance applied so the on-screen view is self-documenting.
    abs_tol = summary.get("tolerance_absolute", 0.0)
    rel_tol = summary.get("tolerance_relative_pct", 0.0) * 100
    st.caption(
        f"Tolerance applied: ± ${abs_tol:,.2f} or {rel_tol:.3g}% of the invoice "
        "amount (whichever is greater). All variances are logged in the workpaper."
    )

    # Mandatory methodology disclaimer, mirrored on-screen.
    st.info(
        "ℹ️ **Methodology & limitations:** this is an automated preparer aid for "
        "pre-population — not a final assurance conclusion. All exceptions, "
        "errors, and unrecorded liabilities require human auditor validation."
    )

    st.divider()

    # --- Prominent download button ------------------------------------------
    st.download_button(
        label="⬇️  Download Audit Workpaper (.xlsx)",
        data=report_bytes,
        file_name=REPORT_FILENAME,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )

    # --- On-screen results preview ------------------------------------------
    if results:
        st.subheader("Reconciliation Detail")
        _render_results_table(results)


# ---------------------------------------------------------------------------
# Main application body
# ---------------------------------------------------------------------------
def main() -> None:
    config = render_sidebar()

    st.title(f"🧾 {APP_NAME}")
    st.markdown(
        "Automated **substantive testing** — vouch a client's General Ledger "
        "against their source invoices (bidirectionally, for both existence and "
        "completeness) and produce a styled audit workpaper in seconds."
    )

    # --- Upload section ------------------------------------------------------
    st.subheader("1 · Upload Source Files")
    up_col1, up_col2 = st.columns(2)

    with up_col1:
        ledger_upload = st.file_uploader(
            "📒 General Ledger (.xlsx)",
            type=["xlsx"],
            accept_multiple_files=False,
            help="The client's exported General Ledger workbook.",
        )

    with up_col2:
        invoice_uploads = st.file_uploader(
            "📄 Invoice Documents (PDF / PNG / JPG …)",
            type=SUPPORTED_INVOICE_TYPES,
            accept_multiple_files=True,
            help="Drag-and-drop one or more supporting invoice documents.",
        )

    # --- Execution -----------------------------------------------------------
    st.subheader("2 · Run the Vouching Engine")
    run_clicked = st.button(
        "🚀 Run Automated Ledger-to-Invoice Vouching",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        # Validate uploads only. The API key is a backend concern handled in code
        # via .env — the OCR module resolves it and raises clearly if misconfigured.
        if ledger_upload is None:
            st.error("Please upload a General Ledger (.xlsx) file.")
            st.stop()
        if not invoice_uploads:
            st.error("Please upload at least one invoice document.")
            st.stop()

        status_placeholder = st.empty()
        try:
            with st.spinner("Running the automated ledger-to-invoice vouching…"):
                reconciliation, report_bytes = run_pipeline(
                    ledger_upload=ledger_upload,
                    invoice_uploads=invoice_uploads,
                    target_sheet=config["target_sheet"],
                    vendor_threshold=config["vendor_threshold"],
                    absolute_tolerance=config["absolute_tolerance"],
                    relative_tolerance_pct=config["relative_tolerance_pct"],
                    status_placeholder=status_placeholder,
                )
            status_placeholder.empty()
        except KeyError as exc:
            status_placeholder.empty()
            # Raised by the ledger parser when required columns can't be matched.
            st.error(f"❌ Ledger parsing failed — {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001 - surface any failure cleanly
            status_placeholder.empty()
            st.error(f"❌ Processing failed: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            st.stop()

        # Persist results in session so reruns (e.g. clicking download) keep them.
        st.session_state["reconciliation"] = reconciliation
        st.session_state["report_bytes"] = report_bytes

    # --- Render results if we have any (survives download-button reruns) -----
    if "reconciliation" in st.session_state and "report_bytes" in st.session_state:
        st.divider()
        st.subheader("3 · Results")
        render_dashboard(
            st.session_state["reconciliation"],
            st.session_state["report_bytes"],
        )


if __name__ == "__main__":
    main()
