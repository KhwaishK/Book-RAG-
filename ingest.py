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
from embedder import embed_text

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 150  # overlap between consecutive chunks (preserves context across cuts)


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


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple fixed-size character chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


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
