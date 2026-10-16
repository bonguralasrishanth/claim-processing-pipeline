"""Small environment-based settings module."""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODE = os.getenv("LLM_MODE", "mock").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))

if LLM_MODE not in {"mock", "openai"}:
    raise ValueError("LLM_MODE must be either 'mock' or 'openai'.")
