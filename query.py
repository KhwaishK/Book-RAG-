"""
Natural language query engine over the ingested book.
Retrieval: Supabase pgvector similarity search.
Generation: Groq's free API running openai/gpt-oss-20b.
"""
import os
from dotenv import load_dotenv
from supabase import create_client
from groq import Groq
from embedder import embed_query

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Free, fast, open-weight model hosted on Groq's free tier.
# Swap to "qwen/qwen3-32b" if you want to compare outputs.
LLM_MODEL = "openai/gpt-oss-20b"


def retrieve_chunks(question: str, book_id: str, k: int = 5):
    q_embedding = embed_query(question)
    result = supabase.rpc("match_chunks", {
        "query_embedding": q_embedding,
        "match_count": k,
        "filter_book_id": book_id,
    }).execute()
    return result.data


def build_prompt(question: str, chunks: list):
    context = "\n\n".join(
        f"[Page {c['page_number']}]\n{c['content']}" for c in chunks
    )
    return f"""You are answering questions about a book using ONLY the excerpts below.
Every claim you make must be followed by a page citation in the form (p. X).
If the excerpts don't contain the answer, say so plainly — do not guess.

Excerpts:
{context}

Question: {question}

Answer (with page citations):"""


def query_book(question: str, book_id: str, k: int = 5):
    chunks = retrieve_chunks(question, book_id, k)

    if not chunks:
        return {
            "answer": "No relevant content found for this book. Has it been ingested?",
            "pages_used": [],
            "retrieved_chunks": [],
        }

    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    pages_used = sorted(set(c["page_number"] for c in chunks))

    return {
        "answer": answer,
        "pages_used": pages_used,
        "retrieved_chunks": chunks,
    }


if __name__ == "__main__":
    import sys
    book_id = sys.argv[1] if len(sys.argv) > 1 else "my-book-v1"
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() in ("quit", "exit"):
            break
        result = query_book(q, book_id)
        print("\n--- Answer ---")
        print(result["answer"])
        print(f"\nPages referenced: {result['pages_used']}")
