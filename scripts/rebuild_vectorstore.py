# scripts/rebuild_vectorstore.py
"""
Fixed Vectorstore Builder
Resolves persistence and validation issues
"""
import time
from pathlib import Path
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core.config import settings
from app.rag.embeddings import get_embeddings, get_embedding_dimension
from app.rag.vectorstore import delete_vectorstore, create_vectorstore_from_documents
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_vectorstore_fixed(force: bool = False, verbose: bool = False) -> bool:
    """
    Build vectorstore with improved persistence handling
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

        # Step 1: Process documents
        logger.info("Step 1: Processing documents...")
        docs_path = Path(settings.DOCS_PATH)
        
        if not docs_path.exists():
            logger.error(f"Documents directory not found: {docs_path}")
            return False

        md_files = list(docs_path.glob("*.md"))
        if not md_files:
            logger.error("No markdown files found")
            return False

        logger.info(f"Found {len(md_files)} markdown files")

        # Process files
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        for md_file in md_files:
            logger.info(f"Processing: {md_file.name}")
            try:
                loader = UnstructuredMarkdownLoader(str(md_file))
                documents = loader.load()
                
                if not documents:
                    logger.warning(f"No content loaded from {md_file}")
                    continue

                # Split documents
                chunks = text_splitter.split_documents(documents)
                
                # Add metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source": md_file.name,
                        "chunk_id": f"{md_file.stem}_{i}",
                        "chunk_index": str(i),
                        "section": "Banking Information",
                        "content_type": "general"
                    })
                
                all_docs.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {md_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")
                continue

        if not all_docs:
            logger.error("No documents were processed successfully")
            return False

        logger.info(f"Total documents processed: {len(all_docs)}")

        # Step 2: Filter metadata
        logger.info("Step 2: Filtering complex metadata...")
        filtered_docs = filter_complex_metadata(all_docs)
        logger.info(f"Documents after filtering: {len(filtered_docs)}")

        # Step 3: Validate embeddings
        logger.info("Step 3: Validating embeddings...")
        embeddings = get_embeddings()
        embedding_dim = get_embedding_dimension()
        logger.info(f"Using embeddings with dimension: {embedding_dim}")

        if verbose:
            try:
                test_embedding = embeddings.embed_query("test document")
                logger.info(f"Embedding test successful - dimension: {len(test_embedding)}")
            except Exception as e:
                logger.warning(f"Embedding test failed: {e}")

        # Step 4: Delete existing vectorstore if needed
        if force and settings.VECTORSTORE_PATH.exists():
            logger.info("Step 4: Removing existing vectorstore...")
            delete_vectorstore()

        # Step 5: Create vectorstore
        logger.info("Step 5: Creating vectorstore...")
        
        # Ensure directory exists
        settings.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        
        vectorstore = create_vectorstore_from_documents(filtered_docs)

        # Step 6: Verify creation
        logger.info("Step 6: Verifying vectorstore creation...")
        
        # Test the vectorstore immediately
        try:
            test_results = vectorstore.similarity_search("banco", k=1)
            if test_results:
                logger.info("✓ Vectorstore search test passed")
            else:
                logger.warning("! No search results returned")
                
            # Check collection count
            if hasattr(vectorstore, '_collection') and vectorstore._collection:
                collection_count = vectorstore._collection.count()
                logger.info(f"Collection reports {collection_count} documents")
                
                if collection_count == len(filtered_docs):
                    logger.info("✓ Document count matches")
                else:
                    logger.warning(f"! Count mismatch: expected {len(filtered_docs)}, got {collection_count}")
            else:
                logger.error("! Could not access collection")
                return False
                
        except Exception as e:
            logger.error(f"Vectorstore verification failed: {e}")
            return False

        # Step 7: Final statistics
        processing_time = time.time() - start_time
        
        # Calculate document statistics
        sections = set()
        content_types = {}
        total_content_length = 0

        for doc in filtered_docs:
            sections.add(doc.metadata.get("section", "Unknown"))
            content_type = doc.metadata.get("content_type", "general")
            content_types[content_type] = content_types.get(content_type, 0) + 1
            total_content_length += len(doc.page_content)

        avg_chunk_size = total_content_length // len(filtered_docs) if filtered_docs else 0

        logger.info("=" * 60)
        logger.info("VECTORSTORE BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"📊 Total chunks: {len(filtered_docs)}")
        logger.info(f"📊 Collection count: {collection_count}")
        logger.info(f"📊 Unique sections: {len(sections)}")
        logger.info(f"📊 Content types: {dict(content_types)}")
        logger.info(f"📊 Avg chunk size: {avg_chunk_size} chars")
        logger.info(f"📊 Embedding dimension: {embedding_dim}")
        logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
        logger.info(f"📍 Location: {settings.VECTORSTORE_PATH}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Vectorstore build failed after {processing_time:.2f}s: {e}")
        return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Build fixed Borgian Bank RAG vectorstore")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if exists")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    success = build_vectorstore_fixed(force=args.force, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
