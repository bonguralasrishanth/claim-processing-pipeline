"""FastAPI entry point for the claim-processing service."""

from __future__ import annotations
from typing import Annotated
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from . import config
from .pdf_utils import PDFProcessingError, extract_pages_from_pdf
from .workflow import process_claim_pages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claim_processing")

app = FastAPI(
    title="Medical Claim Processing Pipeline",
    description=(
        "A FastAPI service that classifies PDF pages and uses a LangGraph workflow "
        "to run three specialized extraction agents."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple service-health endpoint."""
    return {"status": "ok", "llm_mode": config.LLM_MODE}


@app.post("/api/process")
async def process_claim(
    claim_id: Annotated[str, Form(description="Your unique claim identifier")],
    file: Annotated[UploadFile, File(description="A text-based PDF containing claim documents")],
) -> dict:
    """Process an uploaded PDF and return extracted data as JSON.

    The request uses multipart/form-data because it contains both a text field
    (`claim_id`) and a binary file field (`file`).
    """
    if not claim_id.strip():
        raise HTTPException(status_code=422, detail="claim_id must not be empty.")

    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a file with a .pdf extension.")

    pdf_bytes = await file.read()
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The upload is too large. The limit is {config.MAX_UPLOAD_MB} MB.",
        )
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file does not look like a PDF.")

    try:
        pages = extract_pages_from_pdf(pdf_bytes, max_pages=config.MAX_PDF_PAGES)
    except PDFProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    try:
        # The graph uses synchronous nodes; put invocation in a thread so it does
        # not block FastAPI's event loop while an LLM request is running.
        result = await run_in_threadpool(
            process_claim_pages,
            claim_id.strip(),
            pages,
            config.LLM_MODE,
        )
        return result
    except RuntimeError as exc:
        if "OPENAI_API_KEY" in str(exc):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.exception("A workflow runtime error occurred for claim_id=%s", claim_id.strip())
        raise HTTPException(
            status_code=502,
            detail="The claim-processing workflow failed. Check the server terminal for details.",
        ) from exc
    except Exception as exc:
        # Deliberately do not log the raw PDF or extracted medical text.
        logger.exception("Claim processing failed for claim_id=%s", claim_id.strip())
        raise HTTPException(
            status_code=500,
            detail="The claim could not be processed. Check the server terminal for details.",
        ) from exc
