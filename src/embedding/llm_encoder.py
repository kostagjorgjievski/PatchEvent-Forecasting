# src/embedding/llm_encoder.py

from sentence_transformers import SentenceTransformer

class LLMEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
