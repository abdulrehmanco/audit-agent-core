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

# Canonical empty result. Returned (or partially filled) whenever extraction
# cannot produce a value, so downstream code always sees the same shape.
EMPTY_INVOICE: dict[str, Any] = {
    "vendor_name": None,
    "invoice_number": None,
    "date": None,
    "total_amount": None,
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
    '  - "vendor_name": the name of the company that ISSUED the invoice '
    "(the supplier/seller), as a string.\n"
    '  - "invoice_number": the invoice identifier/reference, as a string '
    "(preserve any letters, dashes, and leading zeros).\n"
    '  - "date": the invoice date as a string, exactly as written on the '
    "document.\n"
    '  - "total_amount": the final total amount due as a number (float). '
    "Strip any currency symbols, thousands separators, and whitespace. "
    "Do NOT return a string for this field.\n\n"
    "Rules:\n"
    "  1. Return ONLY the JSON object. No prose, no explanations, no markdown, "
    "no ```json code fences.\n"
    "  2. If a field cannot be found or is ambiguous, set its value to null.\n"
    "  3. The JSON keys must be exactly: vendor_name, invoice_number, date, "
    "total_amount.\n"
    "  4. total_amount must be a JSON number or null — never a quoted string.\n\n"
    "Output schema (shape only):\n"
    '{"vendor_name": "string or null", "invoice_number": "string or null", '
    '"date": "string or null", "total_amount": 0.0}'
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

    for key in ("vendor_name", "invoice_number", "date"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        result[key] = text or None

    # total_amount -> float | None, tolerating "$1,234.50"-style strings.
    amount = payload.get("total_amount")
    if amount is not None:
        if isinstance(amount, (int, float)):
            result["total_amount"] = float(amount)
        else:
            cleaned = "".join(ch for ch in str(amount) if ch.isdigit() or ch in ".-")
            try:
                result["total_amount"] = float(cleaned) if cleaned not in ("", ".", "-") else None
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

    # --- 2. Structured extraction via Groq -----------------------------------
    client = Groq(api_key=groq_api_key)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.0,  # deterministic extraction
            response_format={"type": "json_object"},  # force a JSON object back
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Extract the invoice fields from the following raw "
                        "document text and return ONLY the JSON object:\n\n"
                        "-----BEGIN DOCUMENT-----\n"
                        f"{raw_text}\n"
                        "-----END DOCUMENT-----"
                    ),
                },
            ],
        )
    except Exception as exc:  # noqa: BLE001 - inspect, log, and degrade gracefully.
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        # Groq/OpenAI-style SDKs expose 429 either as a typed RateLimitError or
        # via a status_code attribute; check both the type name and the code.
        is_rate_limited = status == 429 or "ratelimit" in type(exc).__name__.lower()
        if is_rate_limited:
            logger.warning(
                "Groq API rate limit hit (HTTP 429) while processing %s. "
                "Downstream loop should back off and retry. Detail: %s",
                file_path,
                exc,
            )
        else:
            logger.error("Groq API call failed for %s: %s", file_path, exc)
        # Return the empty shape flagged as a TOOLING failure so the match engine
        # surfaces it as PROCESSING_ERROR rather than a false MISSING_DOC.
        failed = dict(EMPTY_INVOICE)
        failed["processing_error"] = True
        failed["error_detail"] = (
            f"Groq API rate limit (HTTP 429): {exc}"
            if is_rate_limited
            else f"Groq API call failed: {type(exc).__name__}: {exc}"
        )
        return failed

    content = completion.choices[0].message.content
    result = _parse_model_json(content)

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

    # 2) Show the full live path *only* if real credentials & file are provided
    #    via environment variables; otherwise just describe what would happen.
    demo_key = os.environ.get("GROQ_API_KEY")
    demo_file = os.environ.get("AUDIT_DEMO_INVOICE")
    if demo_key and demo_file:
        print(f"[demo] Live call: extracting {demo_file} via Groq {GROQ_MODEL} ...\n")
        live_result = extract_invoice_data(demo_file, demo_key)
        print(f"  Result -> {live_result}")
    else:
        print(
            "[demo] Set GROQ_API_KEY and AUDIT_DEMO_INVOICE env vars to run a real\n"
            "       end-to-end extraction. With a fake key the call would hit the\n"
            f"       Groq endpoint and surface an auth error, while a 429 would be\n"
            "       caught and logged as a backoff warning.\n"
        )

    print("[demo] Module is wired correctly. [OK]")
