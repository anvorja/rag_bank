# scripts/rebuild_vectorstore.py
"""
Advanced Vectorstore Builder
Optimized document processing with enhanced metadata extraction
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core.config import settings
from app.rag.embeddings import get_embeddings, get_embedding_dimension
from app.rag.vectorstore import delete_vectorstore, create_vectorstore_from_documents, get_vectorstore_stats
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Advanced document processor with banking-specific optimizations
    """

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.processed_docs = []

        # Initialize text splitters
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ],
            strip_headers=False
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def extract_banking_metadata(self, content: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract banking-specific metadata from document content
        """
        metadata = {
            "section": headers.get("H1", "General"),
            "subsection": headers.get("H2", "N/A"),
            "category": headers.get("H3", "N/A")
        }

        # Banking-specific keyword extraction
        banking_keywords = {
            "productos": ["cuenta", "tarjeta", "crédito", "préstamo", "hipoteca", "inversión"],
            "servicios": ["transferencia", "pago", "consulta", "retiro", "depósito"],
            "canales": ["digital", "móvil", "sucursal", "cajero", "portal"],
            "seguridad": ["clave", "token", "biometría", "seguridad", "verificación"]
        }

        found_keywords = {}
        content_lower = content.lower()

        for category, keywords in banking_keywords.items():
            found = [kw for kw in keywords if kw in content_lower]
            if found:
                found_keywords[category] = found

        # FIXED: Serializar metadatos complejos a strings
        metadata["banking_keywords"] = json.dumps(found_keywords, ensure_ascii=False)
        metadata["content_type"] = self._classify_content_type(content_lower)
        metadata["priority"] = str(self._calculate_priority(content, headers))

        return metadata

    @staticmethod
    def _classify_content_type(content: str) -> str:
        """Classify document content type"""
        if any(word in content for word in ["requisito", "documento", "necesita"]):
            return "requirements"
        elif any(word in content for word in ["tarifa", "costo", "precio", "comisión"]):
            return "pricing"
        elif any(word in content for word in ["proceso", "paso", "cómo", "procedimiento"]):
            return "procedure"
        elif any(word in content for word in ["beneficio", "ventaja", "característica"]):
            return "features"
        else:
            return "general"

    @staticmethod
    def _calculate_priority(content: str, headers: Dict[str, str]) -> int:
        """Calculate content priority (1-5, where 5 is highest)"""
        priority = 3  # Base priority

        # Boost for important sections
        important_sections = ["productos", "servicios", "seguridad", "requisitos"]
        section = headers.get("H1", "").lower()
        if any(imp in section for imp in important_sections):
            priority += 1

        # Boost for detailed content
        if len(content) > 1000:
            priority += 1

        # Cap at 5
        return min(priority, 5)

    def process_markdown_files(self) -> List[Document]:
        """
        Process all markdown files in the docs directory
        """
        docs_path = Path(settings.DOCS_PATH)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")

        md_files = list(docs_path.glob("*.md"))
        if not md_files:
            logger.warning("No markdown files found in documents directory")
            return []

        logger.info(f"Processing {len(md_files)} markdown files")

        all_docs = []

        for md_file in md_files:
            logger.info(f"Processing: {md_file.name}")
            file_docs = self._process_single_file(md_file)
            all_docs.extend(file_docs)
            logger.info(f"Created {len(file_docs)} chunks from {md_file.name}")

        logger.info(f"Total chunks created: {len(all_docs)}")
        return all_docs

    def _process_single_file(self, file_path: Path) -> List[Document]:
        """Process a single markdown file"""
        try:
            # Load document
            loader = UnstructuredMarkdownLoader(str(file_path))
            documents = loader.load()

            if not documents:
                logger.warning(f"No content loaded from {file_path}")
                return []

            # Split by headers first
            header_docs = self.header_splitter.split_text(documents[0].page_content)

            # Further split large chunks
            final_chunks = []
            for i, doc in enumerate(header_docs):
                # Extract metadata
                metadata = self.extract_banking_metadata(doc.page_content, doc.metadata)
                metadata["source"] = file_path.name
                metadata["source_path"] = str(file_path)

                # Split into smaller chunks if needed
                if len(doc.page_content) > self.chunk_size * 1.5:
                    sub_chunks = self.text_splitter.split_text(doc.page_content)
                    for j, chunk_text in enumerate(sub_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = f"{file_path.stem}_{i}_{j}"
                        chunk_metadata["chunk_index"] = str(j)
                        final_chunks.append(Document(
                            page_content=chunk_text,
                            metadata=chunk_metadata
                        ))
                else:
                    metadata["chunk_id"] = f"{file_path.stem}_{i}"
                    metadata["chunk_index"] = "0"
                    final_chunks.append(Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    ))

            return final_chunks

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []


def build_vectorstore(force: bool = False, verbose: bool = False) -> bool:
    """
    Build vectorstore with enhanced processing and validation

    Args:
        force: Force rebuild even if vectorstore exists
        verbose: Enable verbose logging (CORREGIDO: ahora se usa)

    Returns:
        True if successful, False otherwise
    """
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info("STARTING VECTORSTORE BUILD PROCESS")
        logger.info("=" * 60)

        # Check if vectorstore exists
        if settings.VECTORSTORE_PATH.exists() and not force:
            logger.warning("Vectorstore already exists. Use --force to rebuild")
            return False

        # Initialize processor
        processor = DocumentProcessor()

        # Process documents
        logger.info("Step 1: Processing documents...")
        docs = processor.process_markdown_files()

        if not docs:
            logger.error("No documents were processed successfully")
            return False

        # FIXED: Filtrar metadatos complejos
        logger.info("Step 2: Filtering complex metadata...")
        filtered_docs = filter_complex_metadata(docs)

        logger.info(f"Original docs: {len(docs)}, Filtered docs: {len(filtered_docs)}")

        # Validate embeddings (CORREGIDO: ahora se usa la variable 'embeddings')
        logger.info("Step 3: Validating embeddings...")
        embeddings = get_embeddings()
        embedding_dim = get_embedding_dimension()

        # Usar embeddings para una prueba simple si verbose está activado
        if verbose:
            try:
                test_embedding = embeddings.embed_query("test document")
                logger.info(f"Embedding test successful - dimension: {len(test_embedding)}")
            except Exception as e:
                logger.warning(f"Embedding test failed: {e}")

        logger.info(f"Using embeddings with dimension: {embedding_dim}")

        # Delete existing vectorstore if force rebuild
        if force and settings.VECTORSTORE_PATH.exists():
            logger.info("Step 4: Removing existing vectorstore...")
            delete_vectorstore()

        # Create vectorstore
        logger.info("Step 5: Creating vectorstore...")
        create_vectorstore_from_documents(filtered_docs)

        stats = get_vectorstore_stats()
        collection_count = stats.get("count", 0)

        if collection_count != len(filtered_docs):
            logger.warning(f"Document count mismatch: expected {len(filtered_docs)}, got {collection_count}")

        # Summary statistics
        processing_time = time.time() - start_time

        # Document statistics
        sections = set()
        content_types = {}
        total_content_length = 0

        for doc in filtered_docs:
            sections.add(doc.metadata.get("section", "Unknown"))
            content_type = doc.metadata.get("content_type", "general")
            content_types[content_type] = content_types.get(content_type, 0) + 1
            total_content_length += len(doc.page_content)

        logger.info("=" * 60)
        logger.info("VECTORSTORE BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"📊 Total chunks: {len(filtered_docs)}")
        logger.info(f"📊 Vectorstore count: {collection_count}")
        logger.info(f"📊 Unique sections: {len(sections)}")
        logger.info(f"📊 Content types: {dict(content_types)}")
        logger.info(f"📊 Avg chunk size: {total_content_length // len(filtered_docs)} chars")
        logger.info(f"📊 Embedding dimension: {embedding_dim}")
        logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
        logger.info(f"📍 Location: {settings.VECTORSTORE_PATH}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Vectorstore build failed after {processing_time:.2f}s: {e}")
        return False


def inspect_vectorstore() -> bool:
    """
    Inspect existing vectorstore without modifying it

    Returns:
        True if inspection successful, False otherwise
    """
    try:
        logger.info("=" * 60)
        logger.info("VECTORSTORE INSPECTION")
        logger.info("=" * 60)

        if not settings.VECTORSTORE_PATH.exists():
            logger.error("Vectorstore does not exist")
            return False

        stats = get_vectorstore_stats()

        if stats["status"] != "healthy":
            logger.error(f"Vectorstore is not healthy: {stats}")
            return False

        logger.info(f"📊 Status: {stats['status']}")
        logger.info(f"📊 Total chunks: {stats['count']}")
        logger.info(f"📊 Collection: {stats['collection_name']}")
        logger.info(f"📊 Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"📍 Path: {stats['path']}")

        # Test search functionality
        from app.rag.vectorstore import get_vectorstore
        vectorstore = get_vectorstore()

        test_query = "información sobre cuentas bancarias"
        results = vectorstore.similarity_search(test_query, k=3)

        logger.info(f"\n🔍 Test search results for '{test_query}':")
        for i, doc in enumerate(results, 1):
            logger.info(f"  [{i}] {doc.metadata.get('section', 'N/A')} - {doc.page_content[:100]}...")

        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Build or inspect Bank BorjaM RAG vectorstore")

    parser.add_argument("--force", action="store_true", help="Force rebuild even if exists")
    parser.add_argument("--inspect", action="store_true", help="Inspect existing vectorstore")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.inspect:
        success = inspect_vectorstore()
    else:
        success = build_vectorstore(force=args.force, verbose=args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()