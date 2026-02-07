"""
Unit tests for core/pii_detector.py

Testing Strategy main functions:
- regex_scan(): 5 tests (SSN, email, phone, credit card, context-dependent filtering)
- llm_scan(): 3 tests (valid response, malformed JSON, empty text)
- deduplicate_entities(): 2 tests (duplicate removal, unique preservation)
- scan_document(): 2 tests (with LLM, regex-only fallback)
- find_custom_entities(): 3 tests (basic search, case-insensitive, multiple terms)
- get_affected_pages(): 2 tests (normal, empty)
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.pii_detector import (
    regex_scan,
    llm_scan,
    deduplicate_entities,
    scan_document,
    find_custom_entities,
    get_affected_pages,
)
from core.document_store import Entity


class TestRegexScan(unittest.TestCase):
    """Tests for regex_scan() - HIGH priority (fast PII detection backbone)"""

    def test_detects_ssn(self):
        """Test 1: Detects SSN pattern XXX-XX-XXXX"""
        text = "Employee SSN: 123-45-6789 on file."
        entities = regex_scan(text, page_number=1)

        ssn_entities = [e for e in entities if e.type == "SSN"]
        self.assertEqual(len(ssn_entities), 1)
        self.assertEqual(ssn_entities[0].value, "123-45-6789")
        self.assertEqual(ssn_entities[0].page, 1)
        self.assertEqual(ssn_entities[0].confidence, 0.95)
        self.assertTrue(ssn_entities[0].selected)
        self.assertIsNone(ssn_entities[0].bbox)

    def test_detects_email(self):
        """Test 2: Detects email addresses"""
        text = "Contact us at john.smith@acme.com for details."
        entities = regex_scan(text, page_number=3)

        email_entities = [e for e in entities if e.type == "EMAIL"]
        self.assertEqual(len(email_entities), 1)
        self.assertEqual(email_entities[0].value, "john.smith@acme.com")
        self.assertEqual(email_entities[0].page, 3)

    def test_detects_phone(self):
        """Test 3: Detects phone numbers"""
        text = "Call us at (555) 123-4567 today."
        entities = regex_scan(text, page_number=1)

        phone_entities = [e for e in entities if e.type == "PHONE"]
        self.assertEqual(len(phone_entities), 1)
        self.assertIn("123-4567", phone_entities[0].value)

    def test_detects_credit_card(self):
        """Test 4: Detects credit card numbers"""
        text = "Card ending in 4111-1111-1111-1111 was charged."
        entities = regex_scan(text, page_number=2)

        cc_entities = [e for e in entities if e.type == "CREDIT_CARD"]
        self.assertEqual(len(cc_entities), 1)
        self.assertEqual(cc_entities[0].value, "4111-1111-1111-1111")

    def test_context_dependent_filters_false_positives(self):
        """Test 5: Account number pattern requires context keywords to match"""
        # Without context keywords - should NOT match
        text_no_context = "The reference number is 12345678901234 for your order."
        entities_no_ctx = regex_scan(text_no_context, page_number=1)
        acct_entities = [e for e in entities_no_ctx if e.type == "ACCOUNT_NUMBER"]
        self.assertEqual(len(acct_entities), 0)

        # With context keywords - should match
        text_with_context = "Your bank account number is 12345678901234 on file."
        entities_with_ctx = regex_scan(text_with_context, page_number=1)
        acct_entities = [e for e in entities_with_ctx if e.type == "ACCOUNT_NUMBER"]
        self.assertEqual(len(acct_entities), 1)

    def test_empty_text_returns_empty(self):
        """Test 6: Empty or blank text returns no entities"""
        self.assertEqual(regex_scan("", page_number=1), [])
        self.assertEqual(regex_scan("   ", page_number=1), [])


class TestLLMScan(unittest.TestCase):
    """Tests for llm_scan() - HIGH priority (context-aware detection)"""

    def test_parses_valid_llm_response(self):
        """Test 1: Successfully parses well-formed LLM JSON response"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='{"entities": [{"type": "NAME", "value": "John Smith", "context": "CEO John Smith signed"}]}'
        )

        entities = llm_scan("CEO John Smith signed the report.", page_number=1, llm=mock_llm)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].type, "NAME")
        self.assertEqual(entities[0].value, "John Smith")
        self.assertEqual(entities[0].page, 1)
        self.assertEqual(entities[0].confidence, 0.85)

    def test_handles_malformed_json(self):
        """Test 2: Returns empty list when LLM returns bad JSON"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="This is not valid JSON at all")

        entities = llm_scan("Some text here.", page_number=1, llm=mock_llm)
        self.assertEqual(len(entities), 0)

    def test_empty_text_skipped(self):
        """Test 3: Empty text returns empty without calling LLM"""
        mock_llm = Mock()
        entities = llm_scan("", page_number=1, llm=mock_llm)

        self.assertEqual(len(entities), 0)
        mock_llm.invoke.assert_not_called()


class TestDeduplicateEntities(unittest.TestCase):
    """Tests for deduplicate_entities() - MEDIUM priority"""

    def test_removes_duplicate_keeps_higher_confidence(self):
        """Test 1: Same entity from regex (0.95) and LLM (0.85) - keeps regex version"""
        regex_entity = Entity(type="SSN", value="123-45-6789", page=1, confidence=0.95)
        llm_entity = Entity(type="SSN", value="123-45-6789", page=1, confidence=0.85)

        result = deduplicate_entities([regex_entity, llm_entity])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.95)

    def test_preserves_unique_entities(self):
        """Test 2: Different entities from different methods are all kept"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, confidence=0.95),
            Entity(type="NAME", value="John Smith", page=1, confidence=0.85),
            Entity(type="EMAIL", value="j@test.com", page=2, confidence=0.95),
        ]

        result = deduplicate_entities(entities)
        self.assertEqual(len(result), 3)


class TestScanDocument(unittest.TestCase):
    """Tests for scan_document() - HIGH priority (main entry point)"""

    def test_combines_regex_and_llm_results(self):
        """Test 1: Both regex and LLM results are combined and deduplicated"""
        pages = [
            {"page_number": 1, "text": "SSN: 123-45-6789, CEO John Smith"}
        ]

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='{"entities": [{"type": "NAME", "value": "John Smith", "context": "CEO John Smith"}]}'
        )

        entities = scan_document(pages, llm=mock_llm)

        types_found = {e.type for e in entities}
        self.assertIn("SSN", types_found)
        self.assertIn("NAME", types_found)

    def test_regex_only_when_no_llm(self):
        """Test 2: Works with regex-only when LLM is explicitly None"""
        pages = [
            {"page_number": 1, "text": "Email: test@example.com and SSN: 999-88-7777"}
        ]

        # Mock import_module so get_model() import fails, forcing regex-only
        with patch("importlib.import_module", side_effect=ImportError("no llm-utils")):
            entities = scan_document(pages, llm=None)

        # Should still find regex-detectable PII
        types_found = {e.type for e in entities}
        self.assertIn("SSN", types_found)
        self.assertIn("EMAIL", types_found)


class TestFindCustomEntities(unittest.TestCase):
    """Tests for find_custom_entities() - HIGH priority (user-driven redaction)"""

    def test_finds_exact_term(self):
        """Test 1: Finds exact search term in page text"""
        pages = [
            {"page_number": 1, "text": "Contract with Acme Corp for services."},
            {"page_number": 2, "text": "No relevant info here."},
            {"page_number": 3, "text": "Acme Corp headquarters in NYC."},
        ]

        entities = find_custom_entities(pages, ["Acme Corp"])

        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0].page, 1)
        self.assertEqual(entities[1].page, 3)
        self.assertEqual(entities[0].type, "CUSTOM")
        self.assertEqual(entities[0].confidence, 1.0)

    def test_case_insensitive_search(self):
        """Test 2: Matches regardless of case"""
        pages = [
            {"page_number": 1, "text": "ACME CORP is a leading company. acme corp was founded in 1990."}
        ]

        entities = find_custom_entities(pages, ["Acme Corp"])

        self.assertEqual(len(entities), 2)

    def test_multiple_search_terms(self):
        """Test 3: Finds multiple different terms across pages"""
        pages = [
            {"page_number": 1, "text": "John Smith at Acme Corp signed the deal."}
        ]

        entities = find_custom_entities(pages, ["John Smith", "Acme Corp"])

        values = {e.value for e in entities}
        self.assertIn("John Smith", values)
        self.assertIn("Acme Corp", values)


class TestGetAffectedPages(unittest.TestCase):
    """Tests for get_affected_pages() - MEDIUM priority"""

    def test_returns_unique_sorted_pages(self):
        """Test 1: Returns sorted unique page numbers from entities"""
        entities = [
            Entity(type="SSN", value="111-22-3333", page=3),
            Entity(type="NAME", value="Jane", page=1),
            Entity(type="EMAIL", value="a@b.com", page=3),
            Entity(type="PHONE", value="555-1234", page=7),
        ]

        pages = get_affected_pages(entities)
        self.assertEqual(pages, [1, 3, 7])

    def test_empty_entities_returns_empty(self):
        """Test 2: No entities means no affected pages"""
        self.assertEqual(get_affected_pages([]), [])


if __name__ == "__main__":
    unittest.main()
