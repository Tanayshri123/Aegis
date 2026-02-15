"""
RAG Engine - Retrieval-Augmented Generation for document chat.

Builds a knowledge base from document chunks using vector embeddings,
enabling semantic search and context-aware chat even for large documents.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import pickle

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Conditional imports based on deployment mode
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False


class LocalEmbeddings:
    """
    Wrapper for local sentence-transformers embeddings.
    Compatible with LangChain's embedding interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding model.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
                       - Fast, lightweight (80MB)
                       - 384-dimensional embeddings
                       - Good for sovereign/air-gapped deployments
        """
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        print(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded successfully")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class DocumentRAG:
    """
    Handles document chunking, embedding, and retrieval for chat.

    Features:
    - Semantic chunking with overlap for context preservation
    - Vector database for fast similarity search
    - Support for both cloud (NVIDIA) and local (offline) embeddings
    - Persistent storage for knowledge bases
    """

    def __init__(
        self,
        doc_id: str,
        use_local_embeddings: bool = False,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize RAG engine for a document.

        Args:
            doc_id: Unique document identifier
            use_local_embeddings: If True, use local sentence-transformers
                                 If False, use NVIDIA API (requires key)
            embedding_model: Optional model name override
        """
        self.doc_id = doc_id
        self.vector_store = None
        self.use_local = use_local_embeddings

        # Set up embeddings
        if use_local_embeddings:
            model_name = embedding_model or "all-MiniLM-L6-v2" # default local model, all-MiniLM-L6-v2 model is good for document search tasks and text based clustering
            self.embeddings = LocalEmbeddings(model_name)
            print(f"Using LOCAL embeddings (sovereign mode)")
        else:
            if not NVIDIA_AVAILABLE:
                raise ImportError(
                    "langchain-nvidia-ai-endpoints not installed. "
                    "Run: pip install langchain-nvidia-ai-endpoints"
                )

            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError(
                    "NVIDIA_API_KEY not found. Set in .env or use local embeddings."
                )

            # NVIDIA's embedding model
            model_name = embedding_model or "nvidia/nv-embed-v1"
            self.embeddings = NVIDIAEmbeddings(
                model=model_name,
                api_key=api_key
            )
            print(f"☁️  Using NVIDIA embeddings (cloud mode)")

        # Storage paths
        self.vector_db_path = Path(f"./vector_db/{doc_id}")
        self.metadata_path = self.vector_db_path / "metadata.pkl"

    def build_knowledge_base(
        self,
        pages: List[Dict],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict:
        """
        Build vector database from document pages so that we can do semantic search later

        Args:
            pages: List of page dicts from extract_text_from_pdf()
            chunk_size: Characters per chunk (smaller = more precise retrieval)
            chunk_overlap: Overlap between chunks (maintains context)

        Returns:
            Metadata dict with stats
        """
        print(f"\n🔨 Building knowledge base for doc {self.doc_id}...")

        # 1. Convert to LangChain documents
        documents = []
        total_chars = 0

        for page in pages:
            doc = Document(
                page_content=page["text"],
                metadata={
                    "page_number": page["page_number"],
                    "doc_id": self.doc_id
                }
            )
            documents.append(doc)
            total_chars += len(page["text"])

        print(f"  📄 Loaded {len(documents)} pages ({total_chars:,} characters)")

        # 2. Split into smaller chunks for better retrieval
        # RecursiveCharacterTextSplitter tries to split on:
        # - Paragraphs (\n\n)
        # - Sentences (.)
        # - Words ( )
        # Preserving semantic meaning better than fixed-size splits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)

        # 3. Create embeddings and store in vector DB
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=f"doc_{self.doc_id}",
            persist_directory=str(self.vector_db_path)
        )

        # Persist to disk
        self.vector_store.persist()

        # Save metadata
        metadata = {
            "doc_id": self.doc_id,
            "num_pages": len(documents),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chars": total_chars,
            "use_local_embeddings": self.use_local
        }

        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        return metadata

    def load_knowledge_base(self) -> bool:
        """
        Load existing vector database from disk.

        Returns:
            True if loaded successfully, False if not found
        """
        if not self.vector_db_path.exists():
            return False

        try:
            self.vector_store = Chroma(
                collection_name=f"doc_{self.doc_id}",
                persist_directory=str(self.vector_db_path),
                embedding_function=self.embeddings
            )
            return True

        except Exception as e:
            print(f"⚠️  Failed to load knowledge base: {e}")
            return False

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve MOST relevant chunks for a query using semantic search.

        Able to find the chunks that are most semantically similar to the query, even if they don't contain the exact keywords
        Based on cosine similarity of embeddings, not just keyword matching.

        Args:
            query: User's question or search query
            top_k: Number of chunks to retrieve
            score_threshold: Optional minimum similarity score (0-1)

        Returns:
            List of relevant chunks with metadata and scores
        """
        if not self.vector_store:
            raise ValueError(
                "Knowledge base not loaded. "
                "Call build_knowledge_base() or load_knowledge_base() first."
            )

        # Semantic search with scores
        results = self.vector_store.similarity_search_with_score(
            query,
            k=top_k
        )

        # Format results
        context_chunks = []
        for doc, score in results:
            # Filter by the score of the match(cosine similarity)
            # only including chunks that are above a certain threshold GIVEN the threshold exists
            if score_threshold and score < score_threshold:
                continue

            context_chunks.append({
                "text": doc.page_content,
                "page_number": doc.metadata.get("page_number", "Unknown"),
                "relevance_score": float(score),
                "doc_id": doc.metadata.get("doc_id")
            })

        return context_chunks

    def format_context_for_llm(
        self,
        chunks: List[Dict],
        include_scores: bool = False
    ) -> str:
        """
        Format retrieved chunks into a context string for LLM.

        Args:
            chunks: List of chunks from retrieve_context()
            include_scores: Whether to include relevance scores

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Context {i} - Page {chunk['page_number']}"

            if include_scores:
                header += f" - Relevance: {chunk['relevance_score']:.2f}"

            header += "]"

            context_parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(context_parts)

    def chat(
        self,
        user_question: str,
        llm,
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Answer user question using retrieved context.

        Args:
            user_question: User's question
            llm: LLM instance from get_model()
            top_k: Number of chunks to retrieve
            system_prompt: Optional custom system prompt

        Returns:
            Dict with answer, context used, and metadata
        """
        # 1. Retrieve relevant chunks
        relevant_chunks = self.retrieve_context(user_question, top_k=top_k)

        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in the document to answer your question.",
                "context_used": [],
                "num_chunks": 0
            }

        # 2. Format context
        context = self.format_context_for_llm(relevant_chunks)

        # 3. Build prompt, the inputs being JUST the analyzed docs
        default_system = """You are an AI assistant analyzing a document. Use ONLY the provided context to answer questions.

Rules:
- Answer based solely on the context provided
- Cite page numbers when referencing information
- If the context doesn't contain the answer, say so
- Be concise but thorough
- Do not make up information"""

        prompt = f"""{system_prompt or default_system}

CONTEXT:
{context}

USER QUESTION: {user_question}

ANSWER:"""

        # 4. Get answer from LLM
        response = llm.invoke(prompt)

        # Extract text content (handle different response formats)
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, str):
            answer = response
        else:
            answer = str(response)

        return {
            "answer": answer,
            "context_used": relevant_chunks,
            "num_chunks": len(relevant_chunks),
            "pages_referenced": list(set(c["page_number"] for c in relevant_chunks))
        }

    def get_metadata(self) -> Optional[Dict]:
        """
        Get stored metadata about the knowledge base.

        Returns:
            Metadata dict or None if not found
        """
        if not self.metadata_path.exists():
            return None

        try:
            with open(self.metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load metadata: {e}")
            return None

    def delete_knowledge_base(self):
        """Delete the vector database and metadata."""
        import shutil

        if self.vector_db_path.exists():
            shutil.rmtree(self.vector_db_path)
            print(f"🗑️  Deleted knowledge base for doc {self.doc_id}")


# Convenience function for quick setup
def create_rag_engine(doc_id: str, prefer_local: bool = True) -> DocumentRAG:
    """
    Create a RAG engine with automatic embedding selection.

    Args:
        doc_id: Document identifier
        prefer_local: If True, use local embeddings when available

    Returns:
        Configured DocumentRAG instance
    """
    # Try local first if preferred (for sovereign AI)
    if prefer_local and LOCAL_EMBEDDINGS_AVAILABLE:
        return DocumentRAG(doc_id, use_local_embeddings=True)

    # Fall back to NVIDIA API
    elif NVIDIA_AVAILABLE and os.getenv("NVIDIA_API_KEY"):
        return DocumentRAG(doc_id, use_local_embeddings=False)

    # Try local as fallback
    elif LOCAL_EMBEDDINGS_AVAILABLE:
        print("⚠️  NVIDIA API not configured, using local embeddings")
        return DocumentRAG(doc_id, use_local_embeddings=True)

    else:
        raise RuntimeError(
            "No embedding provider available. Install either:\n"
            "  - sentence-transformers (for local/offline): pip install sentence-transformers\n"
            "  - langchain-nvidia-ai-endpoints (for cloud): pip install langchain-nvidia-ai-endpoints"
        )
