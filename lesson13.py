# app/services/ingestion.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


def load_and_chunk_pdf(
    file_path: Path,
    document_id: str,
) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    for page in pages:
        page.metadata["document_id"] = document_id
        page.metadata["filename"] = file_path.name

        # PyPDFLoader generally stores a zero-based page index.
        page_number = page.metadata.get("page")

        if isinstance(page_number, int):
            page.metadata["page_number"] = page_number + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks