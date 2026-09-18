"""
Ingests a PDF book into Supabase, page-by-page, preserving page numbers
so retrieved chunks can always be cited back to a page.

Usage:
    python ingest.py path/to/book.pdf --book-id my-book-v1
"""
import argparse
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import embed_text

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CHUNK_SIZE = 800     
CHUNK_OVERLAP = 150  

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_pages(pdf_path):
    """Extract text per page, keeping page numbers intact (1-indexed)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text):
    """Semantic-aware chunking via LangChain's RecursiveCharacterTextSplitter."""
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def ingest(pdf_path, book_id, batch_size=50):
    print(f"Extracting text from {pdf_path} ...")
    pages = extract_pages(pdf_path)
    print(f"Found {len(pages)} non-empty pages.")

    rows = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for idx, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            rows.append({
                "book_id": book_id,
                "page_number": page["page_number"],
                "chunk_index": idx,
                "content": chunk,
                "embedding": embedding,
            })

    print(f"Generated {len(rows)} chunks. Uploading to Supabase ...")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table("book_chunks").insert(batch).execute()
        print(f"  inserted {i + len(batch)}/{len(rows)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the book PDF")
    parser.add_argument("--book-id", required=True, help="Unique id for this book")
    args = parser.parse_args()

    ingest(args.pdf_path, args.book_id)
