"""
Unit tests for core/document_store.py

Testing Strategy (MEDIUM PRIORITY):
- create(): 2 tests (basic creation, ID uniqueness)
- add_entities(): 2 tests (add entities, multiple adds)
- get_selected_entities(): 2 tests (filter selected, empty)
- update_entity_selection(): 1 test (toggle selection)

Total: 7 tests for core state management functions
Note: Simple getters/setters (get, set_metadata) not tested - trivial
"""

import unittest
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.document_store import (
    Document,
    DocumentStore,
    DocumentStatus,
    Entity
)


class TestDocumentStore(unittest.TestCase):
    """Tests for DocumentStore class."""

    def setUp(self):
        """Run before each test - create fresh DocumentStore."""
        self.store = DocumentStore()

    def test_create_document(self):
        """Test 1: Happy path - create a new document."""
        # Act: Create document
        doc = self.store.create(
            filename="test.pdf",
            original_path="/path/to/test.pdf"
        )

        # Assert: Document created with correct attributes
        self.assertIsNotNone(doc.doc_id)
        self.assertEqual(doc.filename, "test.pdf")
        self.assertEqual(doc.original_path, "/path/to/test.pdf")
        self.assertEqual(doc.status, DocumentStatus.UPLOADING)
        self.assertIsInstance(doc.created_at, datetime)
        self.assertEqual(doc.entities, [])
        self.assertEqual(doc.page_count, 0)

    def test_create_multiple_documents_unique_ids(self):
        """Test 2: Edge case - multiple docs have unique IDs."""
        # Act: Create multiple documents
        doc1 = self.store.create("doc1.pdf", "/path/1")
        doc2 = self.store.create("doc2.pdf", "/path/2")
        doc3 = self.store.create("doc3.pdf", "/path/3")

        # Assert: All IDs are unique
        ids = {doc1.doc_id, doc2.doc_id, doc3.doc_id}
        self.assertEqual(len(ids), 3, "All document IDs should be unique")

    def test_add_entities_to_document(self):
        """Test 3: Happy path - add entities to document."""
        # Arrange: Create document
        doc = self.store.create("test.pdf", "/path")

        # Create entities
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1),
            Entity(type="EMAIL", value="test@example.com", page=2)
        ]

        # Act: Add entities
        self.store.add_entities(doc.doc_id, entities)

        # Assert: Entities added
        retrieved_doc = self.store.get(doc.doc_id)
        self.assertEqual(len(retrieved_doc.entities), 2)
        self.assertEqual(retrieved_doc.entities[0].type, "SSN")
        self.assertEqual(retrieved_doc.entities[1].type, "EMAIL")

    def test_add_entities_multiple_times(self):
        """Test 4: Special behavior - can add entities multiple times."""
        # Arrange
        doc = self.store.create("test.pdf", "/path")
        entities1 = [Entity(type="SSN", value="111-11-1111", page=1)]
        entities2 = [Entity(type="EMAIL", value="test@test.com", page=2)]

        # Act: Add entities twice
        self.store.add_entities(doc.doc_id, entities1)
        self.store.add_entities(doc.doc_id, entities2)

        # Assert: Both batches added
        retrieved_doc = self.store.get(doc.doc_id)
        self.assertEqual(len(retrieved_doc.entities), 2)

    def test_get_selected_entities(self):
        """Test 5: Happy path - filter only selected entities."""
        # Arrange: Create doc with mixed selection
        doc = self.store.create("test.pdf", "/path")
        entities = [
            Entity(type="SSN", value="111-11-1111", page=1, selected=True),
            Entity(type="EMAIL", value="test@test.com", page=2, selected=False),
            Entity(type="PHONE", value="555-1234", page=3, selected=True)
        ]
        self.store.add_entities(doc.doc_id, entities)

        # Act: Get selected only
        selected = self.store.get_selected_entities(doc.doc_id)

        # Assert: Only selected entities returned
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].type, "SSN")
        self.assertEqual(selected[1].type, "PHONE")

    def test_get_selected_entities_empty(self):
        """Test 6: Edge case - no selected entities."""
        # Arrange: Create doc with all unselected
        doc = self.store.create("test.pdf", "/path")
        entities = [
            Entity(type="SSN", value="111-11-1111", page=1, selected=False),
            Entity(type="EMAIL", value="test@test.com", page=2, selected=False)
        ]
        self.store.add_entities(doc.doc_id, entities)

        # Act
        selected = self.store.get_selected_entities(doc.doc_id)

        # Assert: Empty list
        self.assertEqual(selected, [])

    def test_update_entity_selection(self):
        """Test 7: Special behavior - toggle entity selection."""
        # Arrange: Create doc with entity
        doc = self.store.create("test.pdf", "/path")
        entities = [Entity(type="SSN", value="123-45-6789", page=1, selected=True)]
        self.store.add_entities(doc.doc_id, entities)

        # Act: Deselect the entity
        self.store.update_entity_selection(doc.doc_id, "SSN", "123-45-6789", False)

        # Assert: Entity deselected
        retrieved_doc = self.store.get(doc.doc_id)
        self.assertFalse(retrieved_doc.entities[0].selected)


class TestRAGMetadata(unittest.TestCase):
    """Tests for RAG metadata tracking functions."""

    def setUp(self):
        """Create fresh DocumentStore."""
        self.store = DocumentStore()

    def test_mark_and_check_rag_built(self):
        """Test: Mark RAG as built and verify."""
        # Arrange
        doc = self.store.create("test.pdf", "/path")

        # Act: Mark as built
        self.store.mark_rag_built(doc.doc_id)

        # Assert: Should return True
        self.assertTrue(self.store.is_rag_built(doc.doc_id))

    def test_rag_built_false_by_default(self):
        """Test: RAG not built by default."""
        # Arrange
        doc = self.store.create("test.pdf", "/path")

        # Assert: Should return False
        self.assertFalse(self.store.is_rag_built(doc.doc_id))


# Test runner
if __name__ == '__main__':
    unittest.main(verbosity=2)
