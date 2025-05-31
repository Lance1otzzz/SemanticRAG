import unittest
from rag_system.chunking.chonkie_chunker import ChonkieTextChunker, CHONKIE_AVAILABLE


class TestChunker(unittest.TestCase):
    def test_basic_chunking(self):
        text = "A.\n\nB section paragraph. More text."  # two paragraphs
        chunker = ChonkieTextChunker(chunker_config={"chunk_size": 10, "chunk_overlap": 0})
        chunks = chunker.chunk_text(text, metadata={"title": "doc"})
        self.assertGreaterEqual(len(chunks), 2)
        # check metadata presence
        self.assertIn("paragraph_index", chunks[0].metadata)
        if not CHONKIE_AVAILABLE:
            self.assertEqual(chunks[0].metadata["paragraph_index"], 0)


if __name__ == "__main__":
    unittest.main()
