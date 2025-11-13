# scripts/rebuild_vectorstore.py
import shutil
from pathlib import Path
from typing import List, Generator, Type
import structlog

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    TextLoader
)

from app.rag.vectorstore import get_vectorstore
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


def get_document_loaders() -> Generator[tuple[Path, Type], None, None]:
    """Detecta archivos soportados y retorna clases de loaders"""
    supported_extensions = {
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }

    for file_path in settings.DOCS_FOLDER.glob("**/*"):
        if file_path.suffix.lower() in supported_extensions:
            # RETORNA LA CLASE, NO UNA INSTANCIA
            yield file_path, supported_extensions[file_path.suffix.lower()]


def process_documents() -> List[Document]:
    """Carga y procesa todos los documentos en chunks"""
    all_docs: List[Document] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
    )

    for file_path, loader_class in get_document_loaders():
        try:
            logger.info(f"Processing {file_path.relative_to(settings.DOCS_FOLDER)}")

            # Instanciar el loader correctamente
            loader = loader_class(str(file_path))
            pages = loader.load()

            # Extraer estructura de secciones
            for i, doc in enumerate(pages):
                doc.metadata.update({
                    "source": str(file_path),
                    "chunk_id": f"{file_path.name}_chunk_{i}",
                    "file_type": file_path.suffix.lower(),
                    "section": doc.metadata.get("category", "N/A"),
                    "subsection": doc.metadata.get("subcategory", "N/A"),
                })

            # Split en chunks
            chunks = text_splitter.split_documents(pages)
            all_docs.extend(chunks)

            logger.info(f"  → Created {len(chunks)} chunks from {len(pages)} pages")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue

    return all_docs


def rebuild():
    """Reconstruye completamente el vectorstore"""

    # 1. Backup y limpieza
    if settings.VECTORSTORE_PATH.exists():
        backup_path = settings.VECTORSTORE_PATH.with_suffix(".backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        settings.VECTORSTORE_PATH.rename(backup_path)
        logger.info(f"Backed up existing vectorstore to {backup_path}")

    settings.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Procesar documentos
    logger.info(f"Scanning documents in {settings.DOCS_FOLDER}")
    docs = process_documents()

    if not docs:
        logger.warning(f"No documents found in {settings.DOCS_FOLDER}")
        return

    logger.info(f"Total chunks to index: {len(docs)}")

    # 3. Crear vectorstore e insertar documentos
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)

    logger.info(f"Vectorstore rebuilt at {settings.VECTORSTORE_PATH}")
    logger.info(f"Total chunks indexed: {vectorstore._collection.count()}")


if __name__ == "__main__":
    rebuild()