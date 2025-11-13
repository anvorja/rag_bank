# app/rag/retriever.py
"""
Document Retrieval Layer
Implements optimized retrieval strategies for banking documents
"""
from functools import lru_cache
from typing import List, Dict, Any

from langchain_core.documents import Document

from app.core.config import settings
from app.rag.vectorstore import get_vectorstore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizedRetriever:
    """
    Enhanced retriever with banking-specific optimizations
    Implements Maximum Marginal Relevance (MMR) for diversity
    """

    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.k = settings.RETRIEVER_K
        self.fetch_k = settings.RETRIEVER_FETCH_K
        self.lambda_mult = settings.LAMBDA_MULT

        # Initialize retriever with MMR for better diversity
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": self.k,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult
            }
        )

        logger.info(
            f"Retriever initialized",
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
            search_type="mmr"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User's question

        Returns:
            List of relevant documents
        """
        try:
            # Basic retrieval
            docs = self.retriever.invoke(query)

            if not docs:
                logger.warning(f"No documents retrieved for query", query=query[:100])
                return []

            # Post-process documents
            processed_docs = self._post_process_documents(docs, query)

            logger.info(
                f"Retrieved documents",
                query_length=len(query),
                docs_retrieved=len(processed_docs)
            )

            return processed_docs

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Document retrieval failed: {e}")

    def _post_process_documents(self, docs: List[Document], query: str) -> List[Document]:
        """
        Post-process retrieved documents for better quality

        Args:
            docs: Raw retrieved documents
            query: Original query for context

        Returns:
            Processed documents
        """
        processed = []

        for doc in docs:
            # Ensure minimum content quality
            if len(doc.page_content.strip()) < 50:
                continue

            # Add retrieval metadata
            doc.metadata["retrieval_query"] = query[:100]
            doc.metadata["content_length"] = len(doc.page_content)

            # Calculate simple relevance heuristic
            query_words = set(query.lower().split())
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            doc.metadata["word_overlap"] = overlap

            processed.append(doc)

        # Sort by content quality (longer content often better for banking docs)
        processed.sort(key=lambda x: x.metadata.get("content_length", 0), reverse=True)

        return processed

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        Retrieve documents with similarity scores

        Args:
            query: User's question

        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=self.k
            )

            logger.info(
                f"Retrieved documents with scores",
                query_length=len(query),
                docs_retrieved=len(results)
            )

            return results

        except Exception as e:
            logger.error(f"Scored retrieval failed: {e}")
            raise RuntimeError(f"Scored document retrieval failed: {e}")

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval configuration statistics"""
        return {
            "k": self.k,
            "fetch_k": self.fetch_k,
            "lambda_mult": self.lambda_mult,
            "search_type": "mmr",
            "vectorstore_stats": {
                "total_docs": self.vectorstore._collection.count() if self.vectorstore._collection else 0
            }
        }


class HybridRetriever(OptimizedRetriever):
    """
    Advanced hybrid retriever combining multiple strategies
    Future enhancement for more sophisticated retrieval
    """

    def __init__(self):
        super().__init__()
        # Additional initialization for hybrid approach
        self.keyword_boost = 0.3  # Weight for keyword matching

    def retrieve(self, query: str) -> List[Document]:
        """
        Hybrid retrieval combining vector similarity and keyword matching
        """
        # Start with vector retrieval
        vector_docs = super().retrieve(query)

        # For now, return vector results
        # Future: implement keyword retrieval and fusion
        return vector_docs


@lru_cache(maxsize=1)
def get_retriever() -> OptimizedRetriever:
    """
    Get configured retriever instance with caching

    Returns:
        Configured retriever instance
    """
    return OptimizedRetriever()


def get_hybrid_retriever() -> HybridRetriever:
    """
    Get hybrid retriever for advanced use cases

    Returns:
        Configured hybrid retriever instance
    """
    return HybridRetriever()


def test_retriever() -> bool:
    """Test retriever functionality"""
    try:
        retriever = get_retriever()

        # Test queries for banking domain
        test_queries = [
            "información sobre cuentas de ahorro",
            "requisitos para crédito hipotecario",
            "servicios de banca digital"
        ]

        for query in test_queries:
            docs = retriever.retrieve(query)
            if not docs:
                logger.warning(f"No results for test query: {query}")
                return False

        logger.info("Retriever test successful")
        return True

    except Exception as e:
        logger.error(f"Retriever test failed: {e}")
        return False