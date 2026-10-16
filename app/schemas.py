"""Pydantic schemas used to request structured LLM outputs."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

DocumentType = Literal[
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]


class PageClassification(BaseModel):
    page_number: int = Field(description="One-based page number in the source PDF")
    document_type: DocumentType
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
    reason: str = Field(description="Short evidence-based explanation for the classification")


class SegregationResult(BaseModel):
    pages: list[PageClassification]


class BankDetails(BaseModel):
    bank_name: str | None = None
    account_holder_name: str | None = None
    account_numbers: list[str] = Field(default_factory=list)
    cheque_number: str | None = None
    routing_number: str | None = None
    swift_code: str | None = None
    account_type: str | None = None


class IdentityExtraction(BaseModel):
    patient_name: str | None = None
    date_of_birth: str | None = None
    id_numbers: list[str] = Field(default_factory=list)
    claim_reference: str | None = None
    policy_number: str | None = None
    insurance_provider: str | None = None
    bank_details: BankDetails | None = None
    source_pages: list[int] = Field(default_factory=list)


class DischargeExtraction(BaseModel):
    patient_name: str | None = None
    medical_record_number: str | None = None
    diagnosis: str | None = None
    admission_date: str | None = None
    discharge_date: str | None = None
    physician_name: str | None = None
    discharge_condition: str | None = None
    source_pages: list[int] = Field(default_factory=list)


class BillLineItem(BaseModel):
    date: str | None = None
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float


class BillDocument(BaseModel):
    page_number: int
    document_number: str | None = None
    document_type: str | None = None
    items: list[BillLineItem] = Field(default_factory=list)
    reported_subtotal: float | None = None
    tax: float | None = None
    reported_total: float | None = None
    insurance_payment: float | None = None
    patient_responsibility: float | None = None
    calculated_line_item_total: float = 0.0
    totals_match: bool | None = None


class ItemizedBillExtraction(BaseModel):
    documents: list[BillDocument] = Field(default_factory=list)
    combined_line_item_total: float = 0.0
    source_pages: list[int] = Field(default_factory=list)
