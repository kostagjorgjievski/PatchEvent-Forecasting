# tests/test_llm_encoder.py

from sentence_transformers import SentenceTransformer

def test_embedding_shape():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sample = ["bridge holiday | oil shock"]
    emb = model.encode(sample)
    assert emb.shape == (1, 384), f"Expected (1, 384), got {emb.shape}"
