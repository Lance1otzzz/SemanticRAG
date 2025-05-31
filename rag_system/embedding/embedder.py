"""Embedding utilities for the RAG system.

The module primarily exposes :class:`TextEmbedder` which wraps either
``sentence-transformers`` or OpenAI embeddings.  When these libraries are not
available (e.g. in constrained test environments) a simple hashing based
embedding is used so that unit tests can still run without network access.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import json
import hashlib

# ``sentence-transformers`` is an optional dependency.  We try to import it but
# fall back to ``None`` if not installed.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    print("Warning: sentence-transformers not installed. Using hash embeddings for tests.")
    SentenceTransformer = None


OPENAI_API_KEY_PLACEHOLDER = "YOUR_OPENAI_API_KEY_HERE"


class TextEmbedder:
    """Generate embeddings using different backends."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_service: str = "sentence-transformers", api_key: Optional[str] = None) -> None:
        self.model_name = model_name
        self.embedding_service = embedding_service.lower()
        self.client = None
        self.api_key = api_key

        if self.embedding_service == "sentence-transformers":
            if SentenceTransformer is None:
                print("Warning: sentence-transformers not installed. Using hash embeddings for tests.")
            else:
                try:
                    self.client = SentenceTransformer(model_name)
                except Exception as e:  # pragma: no cover - model download may fail
                    print(
                        f"Warning: failed to load SentenceTransformer model '{model_name}', "
                        f"falling back to hash embeddings ({e})."
                    )
                    self.client = None
        elif self.embedding_service == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:  # pragma: no cover - optional dependency
                raise ImportError("OpenAI library not found") from e

            self.api_key = api_key if api_key else OPENAI_API_KEY_PLACEHOLDER
            if self.api_key == OPENAI_API_KEY_PLACEHOLDER:
                print("Warning: OpenAI API key not provided; embeddings will not work.")
            else:
                self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(
                f"Unsupported embedding_service: {embedding_service}."
            )

    # ------------------------------------------------------------------
    def _hash_embed(self, text: str) -> List[float]:
        """Return a deterministic pseudo embedding used for tests."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Take first 8 bytes and convert to floats between 0 and 1
        return [b / 255.0 for b in digest[:8]]

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []

        if self.embedding_service == "sentence-transformers":
            if self.client is not None:
                embeddings = self.client.encode(texts, convert_to_numpy=True)  # pragma: no cover - requires model
                return embeddings.tolist()
            else:
                # Fallback hashing based embeddings
                return [self._hash_embed(t) for t in texts]
        elif self.embedding_service == "openai":
            if self.client is None:
                print("OpenAI client not initialised; returning None")
                return None
            try:  # pragma: no cover - actual API not called in tests
                response = self.client.embeddings.create(input=texts, model=self.model_name)
                return [item.embedding for item in response.data]
            except Exception as e:  # pragma: no cover
                print(f"Error during OpenAI embedding: {e}")
                return None
        return None

    def get_embedding_dimension(self) -> Optional[int]:
        if self.embedding_service == "sentence-transformers" and self.client:
            try:
                return self.client.get_sentence_embedding_dimension()  # pragma: no cover
            except Exception:  # pragma: no cover
                pass
        elif self.embedding_service == "openai" and self.client:
            if self.model_name == "text-embedding-ada-002":
                return 1536
        if self.embedding_service == "sentence-transformers" and self.client is None:
            return len(self._hash_embed("dummy"))
        return None

    # ------------------------------------------------------------------
    def create_embedding_records(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """Return embeddings paired with their metadata."""
        embeddings = self.embed_texts(texts)
        if embeddings is None:
            return None
        if metadatas is None:
            metadatas = [{} for _ in texts]
        records = []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            record = {
                "text": text,
                "embedding": emb,
                "metadata": meta,
            }
            records.append(record)
        return records

    def save_records_jsonl(self, path: str, records: List[Dict[str, Any]]) -> None:
        """Save embedding records to a JSON Lines file."""
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Example usage for manual testing
if __name__ == "__main__":  # pragma: no cover
    embedder = TextEmbedder()
    recs = embedder.create_embedding_records(["hello", "world"], [{"id": 1}, {"id": 2}])
    print(recs)
