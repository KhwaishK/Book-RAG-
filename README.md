# 📖 Book RAG — Natural Language Query System with Page Citations

A Retrieval-Augmented Generation (RAG) system that lets you ask natural
language questions about a ~300-page book and get answers grounded in the
actual text, with every claim traceable back to the exact page it came from.

Built as a submission for [course/assignment name] — see [Project Brief](#project-brief) below.

---

## Demo

Ask a question like *"What is Corporate Social Responsibility (CSR)?"* and get:

> Corporate Social Responsibility (CSR) is the ethical and, in some cases,
> legal duty of a corporation to act responsibly toward its stakeholders...
> (p. 156). CSR also extends beyond the workplace... (p. 100).
>
> 📌 Pages referenced: 100, 156, 248, 275, 353

If the book genuinely doesn't cover something asked about, the system says
so explicitly instead of guessing — e.g. asking about "the fraud triangle"
(a forensic-accounting term not covered in this ethics textbook) correctly
returns *"the excerpts do not contain an explanation of the fraud triangle"*
rather than fabricating an answer.

---

## Project brief

- **Input**: a book of ~300 pages (PDF)
- **Task**: build a natural language query system over it, using RAG
- **Requirement**: every answer must cite the page number(s) it came from
- **Database**: Supabase (Postgres)

## Book used

**Business Ethics** by OpenStax (Rice University), ~375 pages.
Free, open-access, CC BY 4.0 licensed — downloaded from
[openstax.org/details/books/business-ethics](https://openstax.org/details/books/business-ethics).
Included in this repo as `business_ethics.pdf` under the same license
(attribution: *"Access for free at openstax.org"*).

---

## Tech stack (100% free to run)

| Component | Choice | Why |
|---|---|---|
| Vector database | **Supabase** (Postgres + pgvector extension) | Free tier, managed Postgres, native vector similarity search |
| Embedding model | **`BAAI/bge-small-en-v1.5`** via `sentence-transformers` | Runs locally, no API key, no cost, 384-dim, strong retrieval quality for its size |
| LLM (answer generation) | **`openai/gpt-oss-20b`** via [Groq](https://groq.com) free-tier API | Open-weight model, fast inference, generous free rate limits, no local GPU needed |
| PDF parsing | **PyMuPDF (`fitz`)** | Extracts text page-by-page, preserving page numbers |
| Chunking | **LangChain `RecursiveCharacterTextSplitter`** | Splits on paragraph/sentence boundaries first, avoiding mid-sentence cuts |
| UI | **Streamlit** | Minimal code, quick to stand up a usable interface |

Everything above is free — no paid API keys required anywhere in this project.

---

## How it works (architecture)

```
PDF (book)
   ↓ extract text PER PAGE (page numbers preserved from the start)
   ↓ split each page's text into chunks via LangChain's RecursiveCharacterTextSplitter
     (~800 chars, 150 overlap, splitting on paragraph/sentence boundaries where possible)
   ↓ embed each chunk locally (bge-small-en-v1.5)
   ↓ store in Supabase: {book_id, page_number, chunk_index, content, embedding}

User question
   ↓ embed the question (same model)
   ↓ Supabase pgvector cosine-similarity search → top-k matching chunks
   ↓ chunks (still tagged with page numbers) + question → LLM prompt
   ↓ LLM answers, citing (p. X) for every claim, grounded only in retrieved text
```

**Why citations are reliable**: the page number is attached to every chunk
at the moment of extraction — before chunking, embedding, storage, or
retrieval happens. It travels through the entire pipeline as metadata, so
the final citation is a direct lookup, not an inference or guess by the LLM.

**Why it doesn't hallucinate on missing content**: the prompt instructs the
LLM to answer *only* from the retrieved excerpts and to say so plainly if
the answer isn't present, rather than filling gaps from its own training
knowledge.

---

## Project structure

```
book-rag/
├── README.md
├── .env.example         # Template for required secrets (no real keys)
├── .gitignore
├── requirements.txt
├── sql/
│   └── schema.sql        # Supabase table, vector index, match_chunks RPC function
├── embedder.py            # Free local embedding model wrapper
├── ingest.py               # PDF → per-page text → chunks → embeddings → Supabase
├── query.py                # Retrieval (Supabase) + cited answer generation (Groq)
├── app.py                  # Streamlit UI
└── business_ethics.pdf     # Source book (OpenStax, CC BY 4.0)
```

---

## Setup & running it yourself

### Prerequisites
- Python 3.12 (3.10+ should also work)
- A free [Supabase](https://supabase.com) account
- A free [Groq](https://console.groq.com) account (for the LLM)

### 1. Clone and create a virtual environment
```bash
git clone <https://github.com/KhwaishK/Book-RAG->
cd book-rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Set up Supabase
1. Create a new project at [supabase.com](https://supabase.com) (Free plan).
2. Open **SQL Editor** → paste the contents of `sql/schema.sql` → Run.
   This enables `pgvector`, creates the `book_chunks` table, an index, and
   the `match_chunks` similarity-search function.
3. Go to **Project Settings → API Keys → Legacy** tab, copy the
   **Project URL** and the **`service_role`** key.

### 3. Set up Groq (free LLM)
1. Sign up at [console.groq.com](https://console.groq.com).
2. Create an API key under **API Keys**.

### 4. Configure environment variables
Copy `.env.example` to `.env` and fill in your real values:
```
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-api-key
```

### 5. Ingest the book
```bash
python ingest.py business_ethics.pdf --book-id business-ethics-v1
```
This extracts, chunks, embeds, and uploads the book. Takes a few minutes on
CPU for a ~350-page book.

### 6. Run the app
```bash
streamlit run app.py
```
Set **Book ID** in the sidebar to `business-ethics-v1`, ask a question, and
check the cited page numbers against the "View retrieved source chunks"
section.

Or query from the command line instead of the UI:
```bash
python query.py business-ethics-v1
```

---

## Design decisions & limitations

- **Text-only, not multimodal.** The book has no diagrams essential to its
  content, and multimodal support wasn't a stated requirement — kept out of
  scope to avoid overengineering. Could be extended by rendering page images
  and passing them to a multimodal model.
- **Chunking** uses LangChain's `RecursiveCharacterTextSplitter` (~800 chars,
  150 overlap), which splits on paragraph and sentence boundaries where
  possible rather than cutting at a fixed character count regardless of
  context.
- **Retrieval can occasionally miss a relevant section** if the question's
  phrasing differs a lot from the book's own wording (a known limitation of
  embedding-based search generally). Increasing `k` (chunks retrieved) or
  rephrasing closer to the book's terminology usually resolves this.
- **No hallucination on missing content** — if the retrieved chunks don't
  contain the answer, the system says so instead of guessing, verified
  during testing (e.g. a question about "the fraud triangle," a term not
  covered in this book, correctly returned a "not found" response).

## Possible future improvements
- Re-ranking retrieved chunks before passing to the LLM
- Support for multiple books / a book selector in the UI
- Multimodal support for books with meaningful diagrams or figures
- Deploy the Streamlit app (e.g. Streamlit Community Cloud) for a live demo link

## License
Code in this repository is provided for academic/educational submission.
`business_ethics.pdf` is © OpenStax / Rice University, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use,
redistribute, and adapt with attribution.
