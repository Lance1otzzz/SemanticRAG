from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from chonkie import RecursiveChunker
    from chonkie.types import Chunk as ChonkieDocument
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    # 创建一个简单的替代类，用于回退实现
    class ChonkieDocument:
        def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
            self.content = content
            self.metadata = metadata or {}
            print('Chonkie is not imported.')



class ChonkieTextChunker:
    """Utility for splitting text into chunks and capturing paragraph metadata."""

    def __init__(self, chunker_name: str = "RecursiveChunker", chunker_config: Optional[Dict[str, Any]] = None) -> None:
        if chunker_config is None:
            chunker_config = {}
        self.chunker_name = chunker_name
        self.chunker_config = chunker_config

        if CHONKIE_AVAILABLE:
            if chunker_name != "RecursiveChunker":
                raise ValueError(
                    f"Only 'RecursiveChunker' is supported when using the chonkie backend. Got {chunker_name}"
                )
            # 移除 chunk_overlap 参数，因为 RecursiveChunker 不接受该参数
            chunker_params = chunker_config.copy()
            if "chunk_overlap" in chunker_params:
                chunker_params.pop("chunk_overlap")
            self.chunker = RecursiveChunker(**chunker_params)
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
            # 根据新的 API，直接调用 chunker 对象
            chunks = self.chunker(text)
            # 添加元数据
            for idx, chunk in enumerate(chunks):
                # 检查 chunk 对象是否有 metadata 属性，如果没有则添加
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                else:
                    chunk.metadata = chunk.metadata or {}
                chunk.metadata.setdefault("chunk_index", idx)
                # 合并用户提供的元数据
                if metadata:
                    chunk.metadata.update(metadata)
            return chunks
        else:
            return self._fallback_chunk(text, metadata)


# Example usage for manual testing
if __name__ == "__main__":  # pragma: no cover
    sample = """First paragraph.\n\nSecond paragraph that is quite a bit longer and will therefore be split """
    chunker = ChonkieTextChunker(chunker_config={"chunk_size": 20, "chunk_overlap": 5})
    for c in chunker.chunk_text(sample, metadata={"source": "test"}):
        print(c)
