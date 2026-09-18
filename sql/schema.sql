-- Enable pgvector extension (Supabase: Database > Extensions > vector)
create extension if not exists vector;

create table if not exists book_chunks (
  id bigserial primary key,
  book_id text not null,
  page_number int not null,
  chunk_index int not null,
  content text not null,
  embedding vector(384),  -- 384 dims = all-MiniLM-L6-v2 / bge-small-en-v1.5
  created_at timestamp default now()
);

-- Speeds up similarity search (rebuild after bulk insert for best results)
create index if not exists book_chunks_embedding_idx
  on book_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create index if not exists book_chunks_book_id_idx on book_chunks (book_id);

-- RPC function used by the query engine for vector similarity search
create or replace function match_chunks (
  query_embedding vector(384),
  match_count int default 5,
  filter_book_id text default null
)
returns table (
  id bigint,
  page_number int,
  content text,
  similarity float
)
language sql stable
as $$
  select
    id,
    page_number,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from book_chunks
  where filter_book_id is null or book_id = filter_book_id
  order by embedding <=> query_embedding
  limit match_count;
$$;
