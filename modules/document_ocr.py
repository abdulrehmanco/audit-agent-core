"""
Audit Agent: Three-Way Matcher
Module 2 — The Document OCR & Structured Extraction Engine
==========================================================

This module turns a raw, unstructured source document (a vendor invoice as a
native PDF, a scanned PDF, or a flat image) into a clean, structured dictionary
that the downstream deterministic comparison engine can match against the
parsed General Ledger from Module 1.

Pipeline
--------
    1. Local text extraction (free / offline):
         a. Try PyMuPDF (``fitz``) to pull any embedded digital text layer.
         b. If that yields nothing (scanned PDF or image), fall back to
            Tesseract OCR (``pytesseract``) by rasterizing pages / loading the
            image directly.
    2. Structured extraction (cloud):
         - Send the raw text to Groq Cloud (``llama-3.3-70b-versatile``) under a
           strict system prompt and JSON-object response format, forcing the
           model to return ONLY a minified JSON object with the four invoice
           fields we care about.

Public entry point:
    extract_invoice_data(file_path: str, groq_api_key: str) -> dict

Dependencies:
    pymupdf (fitz), pytesseract, Pillow, groq

Note:
    Tesseract OCR must be installed on the host system for the image fallback
    to work (the ``pytesseract`` package is only a thin wrapper around the
    ``tesseract`` binary).
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any

# Load variables from a local .env file (if present) into the environment so the
# Groq API key is configured in code/config — never typed into the UI. This is a
# no-op if python-dotenv isn't installed or no .env exists.
try:
    from dotenv import load_dotenv

    # override=True so the current .env always wins over any stale value already
    # present in the environment (e.g. a placeholder loaded on an earlier run).
    load_dotenv(override=True)
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass

# ---------------------------------------------------------------------------
# Optional heavy dependencies are imported lazily/defensively so that importing
# this module never hard-crashes a host that is missing, say, the Tesseract
# binary. The functions that need each library check for it at call time.
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - environment dependent
    fitz = None

try:
    import pytesseract
    from PIL import Image

    # Allow pointing at the Tesseract executable via .env (TESSERACT_CMD) for
    # Windows hosts where it isn't on the system PATH. No-op if unset.
    _tess_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except ImportError:  # pragma: no cover - environment dependent
    pytesseract = None
    Image = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover - environment dependent
    Groq = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("audit_agent.document_ocr")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Automatic model fallback chain. If the primary model is rate-limited (HTTP 429
# — Groq's free-tier daily caps are PER-MODEL) or unavailable (HTTP 404 / model
# decommissioned), the next model is tried transparently, so neither a daily cap
# nor a retired model ID ever stops a run. All must be JSON-mode compatible.
#   - Override the whole chain with GROQ_MODEL_FALLBACKS (comma-separated).
#   - Setting GROQ_MODEL just changes which model is tried FIRST.
_DEFAULT_FALLBACK_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
]


def get_model_chain() -> list[str]:
    """
    Return the ordered list of models to try, primary first.

    The primary is GROQ_MODEL; fallbacks come from GROQ_MODEL_FALLBACKS (if set)
    or the built-in defaults. Duplicates are removed while preserving order.
    """
    fb_env = (os.environ.get("GROQ_MODEL_FALLBACKS") or "").strip()
    fallbacks = (
        [m.strip() for m in fb_env.split(",") if m.strip()]
        if fb_env
        else list(_DEFAULT_FALLBACK_MODELS)
    )
    chain: list[str] = []
    for model in [GROQ_MODEL, *fallbacks]:
        if model and model not in chain:
            chain.append(model)
    return chain


# A placeholder key (from the shipped .env template) must never be treated as a
# real credential — guard against the common "I forgot to fill it in" mistake.
_PLACEHOLDER_KEYS = {"", "your_groq_api_key_here", "gsk_xxx", "changeme"}


def get_groq_api_key() -> str | None:
    """
    Resolve the Groq API key from the environment (.env / OS env).

    Returns the key string, or None if it is unset or still the placeholder.
    """
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    return None if key in _PLACEHOLDER_KEYS else key


def get_tesseract_version() -> str | None:
    """
    Return the installed Tesseract version string, or None if it's unavailable.

    Lets the UI / diagnostics confirm at runtime whether scanned-image OCR will
    work, instead of guessing from build logs.
    """
    if pytesseract is None:
        return None
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001 - binary missing or not on PATH
        return None


# Canonical empty result. Returned (or partially filled) whenever extraction
# cannot produce a value, so downstream code always sees the same shape.
EMPTY_INVOICE: dict[str, Any] = {
    "vendor_name": None,
    "invoice_number": None,
    "date": None,
    "total_amount": None,
    "line_items": None,    # what was purchased (for GL categorization)
    "bill_to_name": None,  # the buyer — captured only to keep it OUT of vendor_name
}

# DPI used when rasterizing a scanned PDF page for Tesseract. 300 is the sweet
# spot between OCR accuracy and speed/memory.
OCR_RENDER_DPI: int = 300

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")
_PDF_EXTENSIONS = (".pdf",)

# Strict instruction set for the extraction model. Kept as a module constant so
# it can be unit-tested and version-controlled independently of the call site.
SYSTEM_PROMPT: str = (
    "You are an advanced document data-extraction bot for a corporate audit "
    "system. You are given the raw, possibly messy OCR/text dump of a single "
    "vendor invoice. Your sole job is to extract exactly four fields and return "
    "them as a strict, minified JSON object.\n\n"
    "Fields to extract:\n"
    '  - "vendor_name": the name of the company that ISSUED / SOLD / SENT the '
    "invoice — the supplier/seller, normally shown in the letterhead at the very "
    "top of the document or after labels like 'From', 'Seller', 'Supplier', "
    "'Remit To', or 'Vendor'.\n"
    "       CRITICAL: this is NOT the customer/recipient. Do NOT use the company "
    "found under 'Bill To', 'Sold To', 'Ship To', 'Invoice To', 'Customer', or "
    "'Buyer' — that is the buyer, not the vendor. If the seller and a Bill-To "
    "company both appear, always choose the seller (issuer).\n"
    "       On SCANNED/OCR text the layout may be jumbled and the Bill-To name "
    "may appear first or near the top — do not be fooled by position. The seller "
    "is the entity issuing the charge; the company immediately after a 'Bill To'/"
    "'Sold To'/'Ship To' label is ALWAYS the buyer and must go in bill_to_name.\n"
    '  - "invoice_number": the invoice identifier/reference, as a string. '
    "Copy it EXACTLY and COMPLETELY, including any leading 'INV-' / 'INV ' prefix "
    "and all letters, dashes, and leading zeros — never drop or shorten the "
    "prefix.\n"
    '  - "date": the invoice date as a string, exactly as written on the '
    "document.\n"
    '  - "total_amount": the final total amount due as a number (float). '
    "Strip any currency symbols, thousands separators, and whitespace. "
    "Do NOT return a string for this field.\n\n"
    '  - "line_items": a short plain-text summary of WHAT was purchased — the '
    "item/service descriptions only (e.g. 'EC2 compute, S3 storage, data "
    "egress'). Join multiple items with commas. Used for expense categorization. "
    "null if none are legible.\n"
    '  - "bill_to_name": the CUSTOMER/recipient company (the one under "Bill To", '
    '"Sold To", "Ship To", "Customer", or "Buyer"), or null. This field exists '
    "ONLY to keep the buyer OUT of vendor_name — extract it separately so you do "
    "not confuse it with the seller.\n\n"
    "Rules:\n"
    "  1. Return ONLY the JSON object. No prose, no explanations, no markdown, "
    "no ```json code fences.\n"
    "  2. If a field cannot be found or is ambiguous, set its value to null.\n"
    "  3. The JSON keys must be exactly: vendor_name, invoice_number, date, "
    "total_amount, line_items, bill_to_name.\n"
    "  4. total_amount must be a JSON number or null — never a quoted string.\n"
    "  5. Extract ONLY from the document text provided below — never invent or "
    "reuse a company name that does not appear in this document.\n"
    "  6. vendor_name and bill_to_name must be DIFFERENT companies. vendor_name "
    "is the seller/issuer; bill_to_name is the buyer. Never put the Bill-To "
    "company in vendor_name.\n\n"
    "Worked example — given an invoice whose header says 'DocuSign, Inc.' and "
    "whose 'Bill To:' block says 'Brightwater Logistics Inc.', the correct output "
    'is vendor_name="DocuSign, Inc." and bill_to_name="Brightwater Logistics '
    'Inc." — NOT the other way around.\n\n'
    "Output schema (shape only):\n"
    '{"vendor_name": "string or null", "invoice_number": "string or null", '
    '"date": "string or null", "total_amount": 0.0, '
    '"line_items": "string or null", "bill_to_name": "string or null"}'
)


# ---------------------------------------------------------------------------
# Local text extraction
# ---------------------------------------------------------------------------
def _extract_text_with_fitz(file_path: str) -> str:
    """
    Pull the embedded digital text layer from a PDF using PyMuPDF.

    Native (born-digital) PDFs carry a selectable text layer; scanned PDFs do
    not, so this returns an empty/whitespace string for them — which is the
    signal the caller uses to fall back to OCR.

    Args:
        file_path: Path to the PDF.

    Returns:
        The concatenated text of all pages (may be empty / whitespace-only).
    """
    if fitz is None:
        logger.warning("PyMuPDF (fitz) is not installed; skipping digital-layer extraction.")
        return ""

    text_parts: list[str] = []
    # ``with`` ensures the document handle is always closed.
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Run Tesseract OCR on raw image bytes.

    Args:
        image_bytes: Encoded image data (PNG/JPEG/etc.).

    Returns:
        The OCR'd text (may be empty).
    """
    if pytesseract is None or Image is None:
        logger.warning("pytesseract/Pillow not installed; cannot OCR image content.")
        return ""

    with Image.open(io.BytesIO(image_bytes)) as img:
        return pytesseract.image_to_string(img).strip()


def _extract_text_with_ocr(file_path: str) -> str:
    """
    OCR fallback for scanned PDFs and flat images using Tesseract.

    For PDFs, each page is rendered to a high-DPI bitmap and OCR'd in turn (this
    is what makes scanned, image-only PDFs readable). For image files, the bytes
    are OCR'd directly.

    Args:
        file_path: Path to the source document.

    Returns:
        The OCR'd text across all pages/the image (may be empty).
    """
    ext = os.path.splitext(file_path)[1].lower()

    # --- Scanned / image-only PDF: rasterize each page, then OCR -------------
    if ext in _PDF_EXTENSIONS:
        if fitz is None:
            logger.warning("PyMuPDF (fitz) is not installed; cannot rasterize PDF for OCR.")
            return ""
        text_parts: list[str] = []
        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
                page_text = _ocr_image_bytes(pixmap.tobytes("png"))
                if page_text:
                    text_parts.append(page_text)
                logger.debug("OCR'd PDF page %d (%d chars).", page_number, len(page_text))
        return "\n".join(text_parts).strip()

    # --- Flat image file -----------------------------------------------------
    if ext in _IMAGE_EXTENSIONS:
        with open(file_path, "rb") as fh:
            return _ocr_image_bytes(fh.read())

    logger.warning("Unsupported file extension for OCR fallback: %s", ext)
    return ""


def _extract_raw_text(file_path: str) -> str:
    """
    Resolve the raw text of a document using the fitz-first, OCR-fallback chain.

    Strategy:
        1. For PDFs, try the digital text layer via PyMuPDF.
        2. If that comes back empty (scanned PDF) — or the input is an image —
           fall back to Tesseract OCR.

    Args:
        file_path: Path to the source document.

    Returns:
        The best raw text we could obtain.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: No usable text could be extracted by either method.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Source document not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    raw_text = ""

    # Step 1: digital text layer (PDFs only).
    if ext in _PDF_EXTENSIONS:
        try:
            raw_text = _extract_text_with_fitz(file_path)
            if raw_text:
                logger.info("Extracted %d chars from digital text layer (fitz).", len(raw_text))
        except Exception as exc:  # noqa: BLE001 - degrade to OCR rather than abort.
            logger.warning("fitz text extraction failed (%s); falling back to OCR.", exc)
            raw_text = ""

    # Step 2: OCR fallback (scanned PDFs or images).
    if not raw_text:
        logger.info("No digital text layer found; routing %s through Tesseract OCR.", ext)
        try:
            raw_text = _extract_text_with_ocr(file_path)
            if raw_text:
                logger.info("Extracted %d chars via OCR.", len(raw_text))
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR extraction failed for %s: %s", file_path, exc)
            raw_text = ""

    if not raw_text:
        raise ValueError(
            f"Could not extract any text from '{file_path}'. The document may be "
            "empty, corrupt, or Tesseract may not be installed on this host."
        )

    return raw_text


# ---------------------------------------------------------------------------
# LLM response handling
# ---------------------------------------------------------------------------
def _coerce_invoice_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a model JSON payload into our exact four-field schema.

    Guarantees the returned dict always has precisely the four expected keys,
    coerces ``total_amount`` to a float (or None), and leaves the rest as clean
    strings (or None). This makes the output deterministic regardless of small
    model deviations.

    Args:
        payload: The parsed JSON object returned by the model.

    Returns:
        A dict matching EMPTY_INVOICE's shape.
    """
    result = dict(EMPTY_INVOICE)

    for key in ("vendor_name", "invoice_number", "date", "line_items", "bill_to_name"):
        value = payload.get(key)
        if value is None:
            continue
        # The model may return line_items as a list — join into a single string.
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v).strip() for v in value if str(v).strip())
        text = str(value).strip()
        result[key] = text or None

    # Safety net: if the model still placed the buyer in vendor_name (vendor ==
    # bill_to), drop the vendor so a wrong value never propagates silently.
    if (
        result["vendor_name"]
        and result["bill_to_name"]
        and result["vendor_name"].strip().lower() == result["bill_to_name"].strip().lower()
    ):
        logger.warning(
            "Vendor equals bill-to (%r); clearing vendor_name to avoid using the buyer.",
            result["vendor_name"],
        )
        result["vendor_name"] = None

    # total_amount -> float | None, tolerating "$1,234.50"-style strings.
    # Always quantize to cents so a clean amount is never carried with binary
    # float noise (e.g. 921.0700000000001) into the downstream comparison.
    amount = payload.get("total_amount")
    if amount is not None:
        if isinstance(amount, (int, float)):
            result["total_amount"] = round(float(amount), 2)
        else:
            cleaned = "".join(ch for ch in str(amount) if ch.isdigit() or ch in ".-")
            try:
                result["total_amount"] = (
                    round(float(cleaned), 2) if cleaned not in ("", ".", "-") else None
                )
            except ValueError:
                logger.warning("Could not coerce total_amount %r to float.", amount)
                result["total_amount"] = None

    return result


def _parse_model_json(content: str) -> dict[str, Any]:
    """
    Parse the model's text response into a dict, tolerating stray formatting.

    Even with response_format=json_object, we defensively strip any accidental
    ```json fences before parsing so a minor model slip never breaks the run.

    Args:
        content: The raw text content of the model message.

    Returns:
        The coerced four-field invoice dict (EMPTY_INVOICE shape on failure).
    """
    text = (content or "").strip()

    # Strip markdown fences if the model disobeyed and added them.
    if text.startswith("```"):
        text = text.strip("`")
        # Drop a leading language hint like "json\n".
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower() in ("json", ""):
                text = rest
        text = text.strip()

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Model did not return valid JSON (%s). Raw content: %r", exc, content)
        return dict(EMPTY_INVOICE)

    if not isinstance(payload, dict):
        logger.error("Model JSON was not an object: %r", payload)
        return dict(EMPTY_INVOICE)

    return _coerce_invoice_schema(payload)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_invoice_data(file_path: str, groq_api_key: str | None = None) -> dict:
    """
    Extract structured invoice metadata from a source document.

    Combines local OCR/text extraction with a Groq-hosted Llama 3.3 model to
    return the four fields the matcher needs.

    Args:
        file_path:
            Path to the invoice document (.pdf, .png, .jpg, .jpeg, .tiff, ...).
        groq_api_key:
            A valid Groq Cloud API key. If omitted (the normal case), it is read
            from the ``GROQ_API_KEY`` environment variable (loaded from .env).

    Returns:
        A dict with exactly these keys::

            {
                "vendor_name": "Acme Corp" | None,
                "invoice_number": "INV-001" | None,
                "date": "2026-01-15" | None,
                "total_amount": 1250.0 | None,
            }

        On a recoverable failure (rate limit, bad JSON, API error) the function
        returns the empty-shaped dict rather than raising, so a batch loop can
        keep going and apply backoff. Unrecoverable input problems still raise.

    Raises:
        FileNotFoundError: The document path does not exist.
        ValueError: No text could be extracted, or no API key is configured.
        RuntimeError: The ``groq`` SDK is not installed.
    """
    # Fall back to the environment (.env) when no key is passed explicitly.
    groq_api_key = groq_api_key or get_groq_api_key()
    if not groq_api_key:
        raise ValueError(
            "No Groq API key configured. Set GROQ_API_KEY in your .env file "
            "(copy .env.example to .env and fill it in)."
        )
    if Groq is None:
        raise RuntimeError("The 'groq' package is not installed. Run: pip install groq")

    # --- 1. Local text extraction (fitz first, OCR fallback) -----------------
    raw_text = _extract_raw_text(file_path)

    # --- 2. Structured extraction via Groq (with automatic model fallback) ----
    client = Groq(api_key=groq_api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract the invoice fields from the following raw document text "
                "and return ONLY the JSON object:\n\n"
                "-----BEGIN DOCUMENT-----\n"
                f"{raw_text}\n"
                "-----END DOCUMENT-----"
            ),
        },
    ]

    chain = get_model_chain()
    completion = None
    used_model = None
    attempt_errors: list[str] = []

    for model in chain:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.0,  # deterministic extraction
                response_format={"type": "json_object"},  # force a JSON object back
                messages=messages,
            )
            used_model = model
            break
        except Exception as exc:  # noqa: BLE001 - inspect, log, try the next model.
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            ename = type(exc).__name__.lower()
            is_rate_limited = status == 429 or "ratelimit" in ename
            is_unavailable = status == 404 or "notfound" in ename
            is_auth = status in (401, 403)
            attempt_errors.append(f"{model}: {type(exc).__name__}: {exc}")

            if is_rate_limited:
                logger.warning(
                    "Model %s rate-limited (HTTP 429) on %s — falling back to the "
                    "next model.", model, file_path,
                )
            elif is_unavailable:
                logger.warning(
                    "Model %s unavailable (HTTP 404) on %s — falling back to the "
                    "next model.", model, file_path,
                )
            else:
                logger.error("Model %s failed on %s: %s", model, file_path, exc)

            # An auth error affects every model — stop trying the rest.
            if is_auth:
                logger.error("Groq auth error (HTTP %s) — aborting fallback chain.", status)
                break
            continue

    if completion is None:
        # Every model in the chain failed -> surface as a TOOLING failure so the
        # match engine reports PROCESSING_ERROR rather than a false MISSING_DOC.
        failed = dict(EMPTY_INVOICE)
        failed["processing_error"] = True
        failed["error_detail"] = (
            "Groq API call failed for all models in the fallback chain "
            f"({' | '.join(attempt_errors)})"
        )
        logger.error("All %d model(s) failed for %s.", len(chain), file_path)
        return failed

    if used_model != chain[0]:
        logger.info("Extracted %s using fallback model %s.", file_path, used_model)

    content = completion.choices[0].message.content
    result = _parse_model_json(content)
    result["model_used"] = used_model  # audit trail / reproducibility

    # If the model returned nothing usable, treat it as a processing failure
    # (needs human review) rather than letting a blank record flow downstream.
    if not any(result.get(k) for k in ("vendor_name", "invoice_number", "total_amount")):
        result["processing_error"] = True
        result["error_detail"] = "AI returned no usable invoice fields from the document text."
        logger.warning("No usable fields extracted from %s; flagged for review.", file_path)
    else:
        result["processing_error"] = False

    logger.info("Structured extraction complete for %s: %s", file_path, result)
    return result


# ---------------------------------------------------------------------------
# Isolated demonstration: shows wiring without requiring a real key/file.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.getLogger("audit_agent.document_ocr").setLevel(logging.INFO)

    # A stand-in for what _extract_raw_text() would hand to the model after
    # reading a messy scanned invoice.
    mock_raw_text = (
        "ACME INDUSTRIAL SUPPLY CO.\n"
        "123 Warehouse Ave, Springfield\n"
        "INVOICE\n"
        "Invoice #: INV-2026-0042\n"
        "Date: 15 Jan 2026\n"
        "Bill To: Globex Corporation\n"
        "------------------------------\n"
        "Steel brackets x100 .......... $850.00\n"
        "Freight ...................... $150.00\n"
        "TOTAL DUE .................... $1,000.00\n"
    )

    print("\n[demo] Raw text that would be sent to Groq:\n")
    print(mock_raw_text)

    # 1) Demonstrate the offline response-parsing path with a simulated model
    #    reply — this proves our JSON enforcement & coercion without a network call.
    print("[demo] Simulating a model reply and running it through _parse_model_json:\n")
    simulated_reply = (
        '{"vendor_name": "ACME INDUSTRIAL SUPPLY CO.", '
        '"invoice_number": "INV-2026-0042", '
        '"date": "15 Jan 2026", '
        '"total_amount": "$1,000.00"}'
    )
    parsed = _parse_model_json(simulated_reply)
    print(f"  Parsed & coerced -> {parsed}")
    assert parsed["total_amount"] == 1000.0, "Currency string must coerce to float."
    assert parsed["vendor_name"] == "ACME INDUSTRIAL SUPPLY CO."
    assert set(parsed.keys()) == set(EMPTY_INVOICE.keys()), "Schema keys must be exact."

    # Prove the markdown-fence stripping is robust, too.
    fenced = '```json\n{"vendor_name": "X", "invoice_number": null, ' \
             '"date": null, "total_amount": 5}\n```'
    assert _parse_model_json(fenced)["vendor_name"] == "X", "Must strip ```json fences."
    assert _parse_model_json("not json at all") == EMPTY_INVOICE, "Bad JSON -> empty shape."
    print("  JSON enforcement / coercion assertions passed.\n")

    # 2) LIVE regression for Fix B (vendor must be the seller, NOT the Bill-To).
    #    Runs only when a real Groq key is configured (.env) and PyMuPDF is present.
    if get_groq_api_key() and fitz is not None:
        import tempfile

        print("[demo] Live Fix-B check: vendor must NOT be the Bill-To company...\n")
        tmp = tempfile.mkdtemp(prefix="ocr_billto_")
        pdf_path = os.path.join(tmp, "docusign_billto.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 90),
            "\n".join([
                "DocuSign, Inc.", "221 Main Street, San Francisco, CA", "",
                "INVOICE", "Invoice #: INV-DS-60540", "Date: 12 Mar 2026", "",
                "Bill To:", "Brightwater Logistics Inc.", "500 Harbor Blvd", "",
                "Description                 Amount",
                "eSignature annual plan, 50 seats   4800.00",
                "----------------------------------",
                "TOTAL DUE                          4800.00",
            ]),
            fontsize=11, fontname="courier",
        )
        doc.save(pdf_path)
        doc.close()

        res = extract_invoice_data(pdf_path)
        print(f"  vendor_name={res['vendor_name']!r}  bill_to_name={res['bill_to_name']!r}")
        assert res["vendor_name"] and "docusign" in res["vendor_name"].lower(), (
            f"Vendor must be the seller (DocuSign), got {res['vendor_name']!r}"
        )
        assert "brightwater" not in (res["vendor_name"] or "").lower(), (
            "Vendor must NOT be the Bill-To company (Brightwater)."
        )
        print("  Fix-B live check passed: seller extracted, bill-to excluded.\n")
    else:
        print(
            "[demo] (Skipping live Fix-B check — set GROQ_API_KEY in .env to run it.)\n"
        )

    print("[demo] Module is wired correctly. [OK]")
