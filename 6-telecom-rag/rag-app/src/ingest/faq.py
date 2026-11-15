"""Ingests data/faq.csv into the 'faq' Chroma collection (ChromaDB container).
Run once (or whenever the CSV changes):
    docker compose run --rm rag-app python -m src.ingest.faq
"""
from __future__ import annotations

"""
Ingests data/faq.csv into the 'faq' Chroma collection (ChromaDB container).
Run once (or whenever the CSV changes):
    docker compose run --rm rag-app python -m src.ingest.faq
"""
import os
from typing import TYPE_CHECKING

try:
    import chromadb  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    chromadb = None  # type: ignore[assignment]

try:
    import pandas as pd  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    pd = None  # type: ignore[assignment]

try:
    from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    Chroma = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

try:
    from langchain_core.documents import Document as _Document  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    _Document = None  # type: ignore[assignment]

from src.remote_embeddings import RemoteEmbeddings

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION  = "faq"
CSV_PATH    = os.path.join("data", "faq.csv")


def load_faq_documents(csv_path: str) -> list[Document]:
    if _Document is None:
        raise ImportError("langchain-core is required to load FAQ documents")
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(_Document(
            page_content=content,
            metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])},
        ))
    return docs


def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)
    print(f"  {len(docs)} FAQ entries loaded.")

    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embeddings = RemoteEmbeddings()

    print(f"Embedding and storing in collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        client=client,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()