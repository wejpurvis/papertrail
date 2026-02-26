"""
Utility module for generating and comparing embeddings
Loading model at the module level means it loads once the app starts and not on every request, which is more efficient. The `embed_text` function can be imported and used in route handlers or background tasks to generate embeddings for paper abstracts or chunks. The `cosine_similarity` function can be used to compare embeddings when implementing search functionality later on.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    # Returens a 384-dimendsional embedding as a list of floats
    return model.encode(text).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Compute cosine similarity between two embeddings
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0  # or raise ValueError, depending on your use case

    return np.dot(a, b) / (norm_a * norm_b)
