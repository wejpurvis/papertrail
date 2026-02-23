"""
Defines chunking function that takes a string and returns a list of chunk strings
"""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Implements fixed size word chunks wiht overlap
    """

    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words) - overlap:
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += step
    return [c for c in chunks if c]
