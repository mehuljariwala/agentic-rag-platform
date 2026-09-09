"""ragkit -- an agentic retrieval-augmented generation platform.

Quick start:

    from ragkit import Document, RAGPipeline

    rag = RAGPipeline()                 # mock provider, no API key needed
    rag.index([Document(id="d1", text="...")])
    answer = rag.answer("what is ...?")
    print(answer.text, answer.citations)
"""

from .pipeline import RAGConfig, RAGPipeline
from .types import Answer, Chunk, Citation, Document, Hit

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "Hit",
    "RAGConfig",
    "RAGPipeline",
    "__version__",
]
