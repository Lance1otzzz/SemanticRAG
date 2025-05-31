import unittest
from rag_system.embedding.embedder import TextEmbedder


class TestEmbedder(unittest.TestCase):
    def test_create_records(self):
        embedder = TextEmbedder()
        records = embedder.create_embedding_records(["hello"], [{"id": 1}])
        self.assertIsNotNone(records)
        self.assertEqual(records[0]["metadata"], {"id": 1})
        self.assertIn("embedding", records[0])


if __name__ == "__main__":
    unittest.main()
