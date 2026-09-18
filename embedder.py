"""
Shared embedding module.
Uses a free, local sentence-transformers model — no API key, no cost.
"""
from sentence_transformers import SentenceTransformer

# bge-small-en-v1.5: 384 dims, strong retrieval quality for its size, fully free/local.
# Swap to "all-MiniLM-L6-v2" if you want something even lighter/faster.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str):
    model = get_model()
    # bge models recommend a query prefix for queries (not for stored documents)
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_query(text: str):
    model = get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()
