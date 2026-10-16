"""PDF validation and page-by-page text extraction."""

from __future__ import annotations

from typing import Any
import fitz  # PyMuPDF


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot safely be read by the pipeline."""


def extract_pages_from_pdf(pdf_bytes: bytes, max_pages: int = 100) -> list[dict[str, Any]]:
    """Extract selectable text from each PDF page.

    Page numbers returned by this function are 1-based, to be easy for a human
    to compare with the PDF viewer.
    """
    if not pdf_bytes:
        raise PDFProcessingError("The uploaded file is empty.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise PDFProcessingError("Password-protected PDFs are not supported.")

            if document.page_count == 0:
                raise PDFProcessingError("The PDF does not contain any pages.")
            if document.page_count > max_pages:
                raise PDFProcessingError(
                    f"This PDF has {document.page_count} pages. The limit is {max_pages}."
                )

            pages: list[dict[str, Any]] = []
            for index, page in enumerate(document):
                page_text = page.get_text("text").strip()
                pages.append({"page_number": index + 1, "text": page_text})
    except PDFProcessingError:
        raise
    except Exception as exc:  # PyMuPDF raises several exception types for malformed PDFs.
        raise PDFProcessingError("The uploaded file could not be opened as a valid PDF.") from exc

    pages_with_text = [page for page in pages if page["text"]]
    if not pages_with_text:
        raise PDFProcessingError(
            "No selectable text was found in this PDF. It may be a scanned/image-only PDF; "
            "OCR support is not included in this starter project."
        )

    return pages
