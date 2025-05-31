"""Simple text chunking utilities used by the RAG system.

This module provides a small wrapper around the optional `chonkie` library.  If
`chonkie` is available it will be used for chunking, otherwise a very basic
paragraph based chunker is used.  Each produced chunk includes metadata about its
position in the original text so that later retrieval steps can make use of the
structure.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Try to import the real chonkie library.  If it is not installed we fall back to
# a minimal implementation so that the rest of the code can still run (e.g. in
# tests on systems without the dependency).
try:
    from chonkie import Chonkie, DefaultChunker
    from chonkie.types import Document as ChonkieDocument
    CHONKIE_AVAILABLE = True
except Exception:  # pragma: no cover - library is optional
    CHONKIE_AVAILABLE = False

    @dataclass
    class ChonkieDocument:  # type: ignore
        """Lightweight replacement used when `chonkie` isn't installed."""

        content: str
        metadata: Dict[str, Any]

        # Some callers expect a ``tokens`` attribute from chonkie documents.
        # It is left empty here.
        tokens: List[str] | None = None


class ChonkieTextChunker:
    """Utility for splitting text into chunks and capturing paragraph metadata."""

    def __init__(self, chunker_name: str = "DefaultChunker", chunker_config: Optional[Dict[str, Any]] = None) -> None:
        if chunker_config is None:
            chunker_config = {}
        self.chunker_name = chunker_name
        self.chunker_config = chunker_config

        if CHONKIE_AVAILABLE:
            if chunker_name != "DefaultChunker":
                raise ValueError(
                    "Only 'DefaultChunker' is supported when using the chonkie backend."\
                )
            self.chunker = Chonkie(chunker=DefaultChunker(**chunker_config))
        else:
            # Parameters used by the simple fallback chunker
            self.chunk_size = int(chunker_config.get("chunk_size", 500))
            self.chunk_overlap = int(chunker_config.get("chunk_overlap", 0))

    def _fallback_chunk(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[ChonkieDocument]:
        """Fallback paragraph/size based chunking used if chonkie isn't installed."""
        if not text:
            return []
        # Identify paragraphs by double newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[ChonkieDocument] = []
        for p_idx, para in enumerate(paragraphs):
            start = 0
            while start < len(para):
                end = min(start + self.chunk_size, len(para))
                chunk_text = para[start:end]
                meta = dict(metadata or {})
                meta.update({
                    "paragraph_index": p_idx,
                    "start_offset": start,
                    "end_offset": end,
                })
                chunks.append(ChonkieDocument(content=chunk_text, metadata=meta))
                if self.chunk_overlap > 0 and end < len(para):
                    start = end - self.chunk_overlap
                else:
                    start = end
        return chunks

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChonkieDocument]:
        """Return a list of chunk documents with paragraph metadata."""
        if not text:
            return []

        if CHONKIE_AVAILABLE:
            doc = ChonkieDocument(content=text, metadata=metadata or {})
            chunks = self.chunker.chunk([doc])
            for idx, chunk in enumerate(chunks):  # type: ignore[attr-defined]
                chunk.metadata = chunk.metadata or {}
                chunk.metadata.setdefault("chunk_index", idx)
            return chunks  # type: ignore[return-value]
        else:
            return self._fallback_chunk(text, metadata)


# Example usage for manual testing
if __name__ == "__main__":  # pragma: no cover
    sample = """First paragraph.\n\nSecond paragraph that is quite a bit longer and will therefore be split """
    chunker = ChonkieTextChunker(chunker_config={"chunk_size": 20, "chunk_overlap": 5})
    for c in chunker.chunk_text(sample, metadata={"source": "test"}):
        print(c)
