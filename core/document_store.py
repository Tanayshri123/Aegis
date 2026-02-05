"""
Document Store - In-memory state management for Aegis.
Actively tracks documents through their lifecycle from upload to redaction, also stores detected entities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import uuid
from datetime import datetime


class DocumentStatus(Enum):
    """Status of a document in the processing pipeline"""
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    READY = "ready"
    REDACTING = "redacting"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Entity:
    """DETECTED pii entity within a document."""
    type: str
    value: str
    page: int
    confidence: float = 1.0
    selected: bool = True  # whether or not include redacted in document
    context: Optional[str] = None
    bbox: Optional[List[float]] = None  # Bounding box [x1, y1, x2, y2] from Nemotron Parse(black box coordinates)
    semantic_type: Optional[str] = None  # Document element type (title, paragraph, table, etc.)


@dataclass
class Document:
    """Represents a document being processed."""
    doc_id: str
    filename: str
    original_path: str
    status: DocumentStatus
    created_at: datetime
    extracted_text: List[dict] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    redacted_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: int = 0
    page_count: int = 0
    metadata: Dict = field(default_factory=dict)  # RAG status, parsing metadata, etc.


class DocumentStore:
    """In-memory storage for document state."""

    def __init__(self):
        self._documents: Dict[str, Document] = {}

    def create(self, filename: str, original_path: str) -> Document:
        """Create a new document record."""
        doc_id = str(uuid.uuid4())[:8]
        doc = Document(
            doc_id=doc_id,
            filename=filename,
            original_path=original_path,
            status=DocumentStatus.UPLOADING,
            created_at=datetime.now()
        )
        self._documents[doc_id] = doc
        return doc

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def update_status(self, doc_id: str, status: DocumentStatus, progress: int = 0) -> None:
        """Update document status and progress."""
        if doc := self._documents.get(doc_id):
            doc.status = status
            doc.progress = progress

    def set_error(self, doc_id: str, error_message: str) -> None:
        """Mark document as errored."""
        if doc := self._documents.get(doc_id):
            doc.status = DocumentStatus.ERROR
            doc.error_message = error_message

    def add_entities(self, doc_id: str, entities: List[Entity]) -> None:
        """Add detected entities to a document."""
        if doc := self._documents.get(doc_id):
            doc.entities.extend(entities)

    def set_extracted_text(self, doc_id: str, pages: List[dict]) -> None:
        """Set the extracted text for a document."""
        if doc := self._documents.get(doc_id):
            doc.extracted_text = pages
            doc.page_count = len(pages)

    def update_entity_selection(self, doc_id: str, entity_type: str, value: str, selected: bool) -> None:
        """Update whether an entity should be redacted."""
        if doc := self._documents.get(doc_id):
            for entity in doc.entities:
                if entity.type == entity_type and entity.value == value:
                    entity.selected = selected
                    break

    def get_selected_entities(self, doc_id: str) -> List[Entity]:
        """Get all entities marked for redaction."""
        if doc := self._documents.get(doc_id):
            return [e for e in doc.entities if e.selected]
        return []

    def delete(self, doc_id: str) -> bool:
        """Remove a document from storage."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def list_all(self) -> List[Document]:
        """List all documents."""
        return list(self._documents.values())

    def set_metadata(self, doc_id: str, key: str, value: any) -> None:
        """Set a metadata value for a document."""
        if doc := self._documents.get(doc_id):
            doc.metadata[key] = value

    def get_metadata(self, doc_id: str, key: str, default: any = None) -> any:
        """Get a metadata value for a document."""
        if doc := self._documents.get(doc_id):
            return doc.metadata.get(key, default)
        return default

    def mark_rag_built(self, doc_id: str) -> None:
        """Mark that RAG knowledge base has been built for this document."""
        self.set_metadata(doc_id, "rag_built", True)

    def is_rag_built(self, doc_id: str) -> bool:
        """Check if RAG knowledge base exists for this document."""
        return self.get_metadata(doc_id, "rag_built", False)


# Global singleton instance
document_store = DocumentStore()
