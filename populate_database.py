"""
Rebuilds the ChromaDB vector database from the PDF manual.
Run ONCE before starting the app:  python populate_database.py
"""
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function
from utils_env import make_openai_client 
import app_config

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def load_documents():
    pdf_path = Path(app_config.DATA_PATH) / "Manual pentru pasii de asamblare.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def add_ids(chunks):
    last_page = None
    page_idx = 0
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        if page != last_page:
            page_idx = 0
            last_page = page
        chunk.metadata["id"] = f"p{page}:{page_idx}"
        page_idx += 1
    return chunks


def main():
    print("Încarc .env și inițializez clientul OpenAI...")
    make_openai_client()  

    print(f"Citesc PDF din: {app_config.DATA_PATH}")
    docs = load_documents()
    print(f"Pagini încărcate: {len(docs)}")

    chunks = split_documents(docs)
    chunks = add_ids(chunks)
    print(f"Chunk-uri create: {len(chunks)}")

    chroma_path = Path(app_config.CHROMA_PATH)
    if chroma_path.exists():
        print(f"Șterg baza de date veche din: {chroma_path}")
        shutil.rmtree(chroma_path)

    print("Calculez embeddings și construiesc baza de date (poate dura 1-2 minute)...")
    db = Chroma.from_documents(
        chunks,
        get_embedding_function(),
        persist_directory=str(chroma_path),
    )
    
    count = db._collection.count()
    print(f"\nGata! {count} chunk-uri salvate în {chroma_path}")
    print("Acum poți rula test_rag.py sau licenta.py")


if __name__ == "__main__":
    main()
