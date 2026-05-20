import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =========================================================
# CONFIG
# =========================================================

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# =========================================================
# LOAD EMBEDDING MODEL ONCE (IMPORTANT)
# =========================================================

_embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)


# =========================================================
# GET EMBEDDINGS
# =========================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    return _embedding_model


# =========================================================
# BUILD VECTOR STORE
# =========================================================

def build_vector_store(transcript: str) -> Chroma:

    print("\n[INFO] Building Vector Store...")

    # Ensure directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # -------------------------
    # TEXT SPLITTING
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(transcript)

    print(f"[INFO] Total Chunks Created: {len(chunks)}")

    # -------------------------
    # CREATE DOCUMENTS
    # -------------------------

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # -------------------------
    # LOAD EMBEDDINGS
    # -------------------------

    embeddings = get_embeddings()

    # -------------------------
    # CREATE VECTOR STORE
    # -------------------------

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print("[INFO] Vector Store Created Successfully!")

    return vector_store


# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store() -> Chroma:

    print("\n[INFO] Loading Existing Vector Store...")

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("[INFO] Vector Store Loaded Successfully!")

    return vector_store


# =========================================================
# GET RETRIEVER
# =========================================================

def get_retriever(vector_store: Chroma, k: int = 4):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )