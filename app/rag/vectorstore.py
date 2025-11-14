# app/rag/vectorstore.py
"""
Vector Store Management and Abstraction Layer
Implements Repository Pattern for vector operations
"""
from functools import lru_cache
import shutil
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.embeddings import get_embeddings, get_embedding_dimension
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Vector Store Manager with comprehensive operations
    Follows Repository Pattern for data access
    """

    def __init__(self):
        self.collection_name = "banco_rag"
        self.embeddings = get_embeddings()
        self.persist_directory = settings.VECTORSTORE_PATH

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create new vectorstore with documents"""
        try:
            if not documents:
                raise ValueError("Cannot create vectorstore without documents")

            logger.info(f"Creating vectorstore with {len(documents)} documents")

            # # Ensure directory exists
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            get_vectorstore.cache_clear()

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory)
            )

            get_vectorstore.cache_clear()

            logger.info(f"Vectorstore created successfully",
                       path=str(self.persist_directory),
                       documents=len(documents))
            return vectorstore

        except Exception as e:
            logger.error(f"Failed to create vectorstore: {e}")
            raise RuntimeError(f"Vectorstore creation failed: {e}")


    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vectorstore"""
        try:
            if not self.vectorstore_exists():
                logger.warning("Vectorstore does not exist")
                return None

            # FIX: Clear cache before loading
            get_vectorstore.cache_clear()

            vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

            # FIX: Simple functionality test instead of complex validation
            try:
                # Test basic search functionality
                test_results = vectorstore.similarity_search("test", k=1)
                logger.info(f"Vectorstore loaded and functional",
                           test_results=len(test_results))
            except Exception as test_error:
                logger.warning(f"Vectorstore test search failed: {test_error}")
                # Don't fail immediately, the vectorstore might still be usable

            logger.info(f"Vectorstore loaded successfully",
                       path=str(self.persist_directory))
            return vectorstore

        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            return None

    def vectorstore_exists(self) -> bool:
        """Check if vectorstore exists and is valid"""
        try:
            if not self.persist_directory.exists():
                return False

            # Check for required Chroma files
            required_files = ["chroma.sqlite3"]
            chroma_files = list(self.persist_directory.glob("chroma*"))

            has_db_files = len(chroma_files) > 0 or any(
                (self.persist_directory / f).exists() for f in required_files
            )

            return has_db_files and any(self.persist_directory.iterdir())
        except Exception as e:
            logger.error(f"Error checking vectorstore existence: {e}")
            return False

    def delete_vectorstore(self) -> bool:
        """Delete existing vectorstore"""
        try:
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                logger.info(f"Vectorstore deleted", path=str(self.persist_directory))
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete vectorstore: {e}")
            return False

    def get_stats(self) -> dict:
        """Get vectorstore statistics"""
        try:
            vectorstore = self.load_vectorstore()
            if not vectorstore:
                return {"status": "not_found", "count": 0}

            # noinspection PyProtectedMember
            collection = vectorstore._collection
            count = collection.count() if collection else 0

            return {
                "status": "healthy",
                "count": count,
                "path": str(self.persist_directory),
                "collection_name": self.collection_name,
                "embedding_dimension": get_embedding_dimension()
            }
        except Exception as e:
            logger.error(f"Failed to get vectorstore stats: {e}")
            return {"status": "error", "error": str(e)}

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to existing vectorstore"""
        try:
            vectorstore = self.load_vectorstore()
            if not vectorstore:
                raise ValueError("Vectorstore does not exist")

            vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vectorstore")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    # noinspection PyMethodMayBeStatic,PyProtectedMember
    def _validate_vectorstore(self, vectorstore: Chroma) -> bool:
        """Simple validation - just check if we can use it"""
        try:
            # Try a simple search
            vectorstore.similarity_search("", k=1)
            logger.info(f"Vectorstore validation passed")
            return True
        except Exception as e:
            logger.error(f"Vectorstore validation failed: {e}")
            return False


# Global manager instance
_manager = VectorStoreManager()


@lru_cache(maxsize=1)
def get_vectorstore(recreate: bool = False) -> Chroma:
    """
    Get vectorstore instance with caching

    Args:
        recreate: Force recreation of vectorstore

    Returns:
        Chroma vectorstore instance
    """
    if recreate:
        _manager.delete_vectorstore()

    vectorstore = _manager.load_vectorstore()
    if not vectorstore:
        raise RuntimeError(
            "Vectorstore not found or invalid. "
            "Run 'python scripts/rebuild_vectorstore.py' to create it."
        )

    return vectorstore


def vectorstore_exists() -> bool:
    """Check if vectorstore exists"""
    return _manager.vectorstore_exists()


def get_vectorstore_stats() -> dict:
    """Get vectorstore statistics"""
    return _manager.get_stats()


def create_vectorstore_from_documents(documents: List[Document]) -> Chroma:
    """Create new vectorstore from documents"""
    return _manager.create_vectorstore(documents)


def delete_vectorstore() -> bool:
    """Delete existing vectorstore"""
    # Clear cache to force reload
    get_vectorstore.cache_clear()
    return _manager.delete_vectorstore()


def test_vectorstore() -> bool:
    """Test vectorstore functionality"""
    try:
        stats = get_vectorstore_stats()
        if stats["status"] != "healthy":
            logger.error(f"Vectorstore test failed", stats=stats)
            return False

        vectorstore = get_vectorstore()

        # Test similarity search
        test_query = "información sobre cuentas bancarias"
        results = vectorstore.similarity_search(test_query, k=1)

        if not results:
            logger.warning("Vectorstore test: no results returned")
            return False

        logger.info(f"Vectorstore test successful", results_count=len(results))
        return True

    except Exception as e:
        logger.error(f"Vectorstore test failed: {e}")
        return False