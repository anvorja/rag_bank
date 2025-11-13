"""
rebuild_vectorstore.py
Reconstruye el vectorstore con soporte para Markdown y PDFs
Optimizado para documentos bancarios estructurados
"""

import os
import shutil
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    RecursiveCharacterTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document

# ================================
# CONFIGURACIÓN
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no está definida en las variables de entorno")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "banco_rag"
DOCS_FOLDER = "./docs"  # Carpeta para Markdown y PDFs

# ================================
# CONFIGURACIÓN DE CHUNKING
# ================================
# Para Markdown: dividir por headers primero
HEADERS_TO_SPLIT = [
    ("#", "seccion"),
    ("##", "subseccion"),
    ("###", "topico"),
]

# Configuración de chunking secundario
CHUNK_SIZE = 800  # Tokens aproximados
CHUNK_OVERLAP = 150  # Overlap para mantener contexto

# ================================
# FUNCIONES DE CARGA
# ================================
def load_markdown_files() -> List[Document]:
    """Carga archivos Markdown con chunking semántico por headers"""
    print("\n📄 Procesando archivos Markdown...")
    docs = []
    
    md_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".md")]
    
    if not md_files:
        print("  ⚠ No se encontraron archivos .md")
        return docs
    
    # Splitter por headers de Markdown
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False  # Mantener headers en el contenido
    )
    
    # Splitter secundario para chunks grandes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        length_function=len,
    )
    
    for file in md_files:
        path = os.path.join(DOCS_FOLDER, file)
        print(f"  📝 Procesando: {file}")
        
        try:
            # Leer contenido
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split por headers (preserva metadata de secciones)
            header_splits = md_splitter.split_text(content)
            
            # Split adicional si los chunks son muy grandes
            final_chunks = []
            for doc in header_splits:
                if len(doc.page_content) > CHUNK_SIZE * 1.5:
                    # Chunk es muy grande, dividir más
                    sub_chunks = text_splitter.split_documents([doc])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(doc)
            
            # Agregar metadata de origen
            for chunk in final_chunks:
                chunk.metadata["source"] = path
                chunk.metadata["file_type"] = "markdown"
                chunk.metadata["source_file"] = file
            
            docs.extend(final_chunks)
            print(f"    ✓ {len(final_chunks)} chunks creados")
            
        except Exception as e:
            print(f"    ✗ Error procesando {file}: {e}")
    
    return docs

def load_pdf_files() -> List[Document]:
    """Carga archivos PDF con chunking tradicional"""
    print("\n📕 Procesando archivos PDF...")
    docs = []
    
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("  ⚠ No se encontraron archivos .pdf")
        return docs
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    for file in pdf_files:
        path = os.path.join(DOCS_FOLDER, file)
        print(f"  📄 Procesando: {file}")
        
        try:
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            chunks = text_splitter.split_documents(raw_docs)
            
            # Agregar metadata
            for chunk in chunks:
                chunk.metadata["source"] = path
                chunk.metadata["file_type"] = "pdf"
                chunk.metadata["source_file"] = file
            
            docs.extend(chunks)
            print(f"    ✓ {len(chunks)} chunks creados")
            
        except Exception as e:
            print(f"    ✗ Error procesando {file}: {e}")
    
    return docs

# ================================
# FUNCIÓN PRINCIPAL
# ================================
def rebuild_vectorstore():
    """Reconstruye completamente el vectorstore"""
    print("\n" + "="*60)
    print("🔄 RECONSTRUCCIÓN DE VECTORSTORE")
    print("="*60)
    
    # Verificar carpeta de documentos
    if not os.path.exists(DOCS_FOLDER):
        print(f"\n❌ Error: La carpeta '{DOCS_FOLDER}' no existe")
        print(f"   Crea la carpeta y agrega tus archivos .md y/o .pdf")
        return
    
    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(('.md', '.pdf'))]
    if not files:
        print(f"\n❌ Error: No hay archivos .md o .pdf en '{DOCS_FOLDER}'")
        return
    
    print(f"\n📂 Archivos encontrados: {len(files)}")
    for f in files:
        print(f"   • {f}")
    
    # Confirmar reconstrucción
    print("\n⚠️  ADVERTENCIA: Esto BORRARÁ el vectorstore actual.")
    confirm = input("¿Estás seguro de continuar? (s/N): ")
    if confirm.lower() != "s":
        print("❌ Operación cancelada.")
        return
    
    # 1. Borrar vectorstore existente
    if os.path.exists(VECTORSTORE_PATH):
        print(f"\n🗑️  Borrando vectorstore existente: {VECTORSTORE_PATH}")
        shutil.rmtree(VECTORSTORE_PATH)
    
    # 2. Crear directorios necesarios
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    
    # 3. Cargar documentos
    all_docs = []
    
    # Cargar Markdown (prioritario)
    md_docs = load_markdown_files()
    all_docs.extend(md_docs)
    
    # Cargar PDFs (si existen)
    pdf_docs = load_pdf_files()
    all_docs.extend(pdf_docs)
    
    if not all_docs:
        print("\n❌ No se pudieron procesar documentos.")
        return
    
    # 4. Estadísticas
    print(f"\n📊 RESUMEN:")
    print(f"   • Total de chunks: {len(all_docs)}")
    print(f"   • Chunks de Markdown: {len(md_docs)}")
    print(f"   • Chunks de PDF: {len(pdf_docs)}")
    
    # Estadísticas de metadata
    sections = set()
    for doc in all_docs:
        if "seccion" in doc.metadata:
            sections.add(doc.metadata["seccion"])
    
    if sections:
        print(f"   • Secciones identificadas: {len(sections)}")
    
    # 5. Crear embeddings y vectorstore
    print(f"\n🔧 Creando embeddings con OpenAI...")
    print(f"   Modelo: text-embedding-3-small")
    
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print(f"   Creando vectorstore en: {VECTORSTORE_PATH}")
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTORSTORE_PATH
        )
        
        print(f"\n✅ VECTORSTORE RECONSTRUIDO EXITOSAMENTE")
        print(f"   📍 Ubicación: {VECTORSTORE_PATH}")
        print(f"   📦 Collection: {COLLECTION_NAME}")
        print(f"   📈 Total chunks indexados: {len(all_docs)}")
        
        # Verificación
        collection_count = vectorstore._collection.count()
        print(f"   ✓ Verificado en Chroma: {collection_count} documentos")
        
    except Exception as e:
        print(f"\n❌ Error creando vectorstore: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 Proceso completado. Puedes iniciar el servidor con:")
    print("   python app.py")
    print("="*60 + "\n")

# ================================
# FUNCIÓN DE INSPECCIÓN
# ================================
def inspect_vectorstore():
    """Inspecciona el vectorstore actual sin modificarlo"""
    print("\n" + "="*60)
    print("🔍 INSPECCIÓN DE VECTORSTORE")
    print("="*60)
    
    if not os.path.exists(VECTORSTORE_PATH):
        print("\n❌ Vectorstore no existe")
        return
    
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embedding,
            collection_name=COLLECTION_NAME
        )
        
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   • Total de chunks: {count}")
        print(f"   • Collection: {COLLECTION_NAME}")
        print(f"   • Ubicación: {VECTORSTORE_PATH}")
        
        # Obtener una muestra
        if count > 0:
            sample = collection.peek(limit=5)
            print(f"\n📝 MUESTRA DE DOCUMENTOS (primeros 5):")
            for i, doc in enumerate(sample['documents'][:5], 1):
                metadata = sample['metadatas'][i-1] if i-1 < len(sample['metadatas']) else {}
                print(f"\n   [{i}] {metadata.get('source_file', 'N/A')}")
                print(f"       Sección: {metadata.get('seccion', 'N/A')}")
                print(f"       Preview: {doc[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error inspeccionando: {e}")
    
    print("\n" + "="*60 + "\n")

# ================================
# EJECUTAR
# ================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect_vectorstore()
    else:
        rebuild_vectorstore()
