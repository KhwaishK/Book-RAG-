from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str):
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_query(text: str):
    model = get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()
