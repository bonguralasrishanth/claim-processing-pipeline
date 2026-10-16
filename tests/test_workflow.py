from app.workflow import (
    _decimal_total,
    _mock_document_type,
    process_claim_pages,
    route_to_extraction_agents,
)


def test_mock_classifier_uses_assignment_categories():
    assert _mock_document_type("MEDICAL CLAIM FORM")[0] == "claim_forms"
    assert _mock_document_type("GOVERNMENT ID CARD")[0] == "identity_document"
    assert _mock_document_type("ITEMIZED HOSPITAL BILL")[0] == "itemized_bill"
    assert _mock_document_type("RANDOM LETTER")[0] == "other"


def test_router_sends_only_assigned_pages_to_each_agent():
    state = {
        "claim_id": "test-1",
        "mode": "mock",
        "pages": [
            {"page_number": 1, "text": "Claim form text"},
            {"page_number": 2, "text": "ID text"},
            {"page_number": 3, "text": "Discharge summary text"},
            {"page_number": 4, "text": "Itemized hospital bill text"},
            {"page_number": 5, "text": "Prescription text"},
        ],
        "classifications": [
            {"page_number": 1, "document_type": "claim_forms"},
            {"page_number": 2, "document_type": "identity_document"},
            {"page_number": 3, "document_type": "discharge_summary"},
            {"page_number": 4, "document_type": "itemized_bill"},
            {"page_number": 5, "document_type": "prescription"},
        ],
    }
    sends = route_to_extraction_agents(state)
    by_node = {send.node: send.arg["pages"] for send in sends}
    assert [p["page_number"] for p in by_node["id_agent"]] == [1, 2]
    assert [p["page_number"] for p in by_node["discharge_summary_agent"]] == [3]
    assert [p["page_number"] for p in by_node["itemized_bill_agent"]] == [4]
    assert all(5 not in [p["page_number"] for p in pages] for pages in by_node.values())


def test_decimal_total_is_rounded_to_cents():
    assert _decimal_total([1.10, 2.20, 0.05]) == 3.35


def test_mock_graph_returns_expected_sections():
    pages = [
        {
            "page_number": 1,
            "text": (
                "MEDICAL CLAIM FORM\nHealthCare Insurance Company\nClaim Reference: CLM-123\n"
                "Patient Name:\nJane Doe\nDate of Birth:\nJanuary 2, 1980\nPolicy Number:\nPOL-123"
            ),
        },
        {
            "page_number": 2,
            "text": (
                "DISCHARGE SUMMARY\nPatient Name:\nJane Doe\nMRN:\nMRN-1\n"
                "Admission Date:\nJanuary 1, 2025\nDischarge Date:\nJanuary 3, 2025\n"
                "Attending Physician:\nDr Example\nAdmission Diagnosis:\nPneumonia"
            ),
        },
        {
            "page_number": 3,
            "text": (
                "ITEMIZED HOSPITAL BILL\nBill Number:\nBILL-123\nITEMIZED CHARGES\nDATE\nDESCRIPTION\n"
                "QTY\nRATE\nAMOUNT\n01/01/25\nRoom charge\n1\n$100.00\n$100.00\n"
                "Subtotal:\n$100.00\nTax (5%):\n$5.00\nTotal Amount:\n$105.00"
            ),
        },
    ]
    result = process_claim_pages("test-claim", pages, "mock")
    assert result["claim_id"] == "test-claim"
    assert result["pipeline_mode"] == "mock"
    assert result["extracted_data"]["identity"]["patient_name"] == "Jane Doe"
    assert result["extracted_data"]["discharge_summary"]["diagnosis"] == "Pneumonia"
    bill_doc = result["extracted_data"]["itemized_bill"]["documents"][0]
    assert bill_doc["calculated_line_item_total"] == 100.0
    assert bill_doc["totals_match"] is True
