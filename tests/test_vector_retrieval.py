import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.services.instruction_library import InstructionLibrary
from test_instruction_workflow import make_pdf


class TestEmbeddings:
    model_name = "test"
    model_id = "test:1"

    def documents(self, texts):
        return [b"test" for _ in texts]

    def query(self, text):
        return SimpleNamespace(nbytes=4)

    def rank(self, query, vectors, floor, limit):
        return [row[0] for row in vectors[:limit]]


class VectorRetrievalTests(unittest.TestCase):
    def test_backfill_hybrid_and_revision(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(settings, "rag_embedding_model", ""):
            path = str(Path(folder) / "test.sqlite3")
            library = InstructionLibrary(path)
            library.ingest("manual.pdf", make_pdf("确认卡扣完整"), "WS1001")
            library.ingest("manual.pdf", make_pdf("记录编号"), "WS1002")
            library.embeddings = TestEmbeddings()
            self.assertEqual(library.vector_status()["indexed"], 0)
            status = library.backfill_vectors(batch_size=1)
            self.assertGreater(status["chunks"], 0)
            self.assertEqual(status["indexed"], status["chunks"])
            results = library.retrieve("卡扣", station="WS1001")
            self.assertIn("卡扣", results[0].page_content)
            self.assertEqual(results[0].metadata["retrieved_by"], ["keyword", "semantic"])
            paraphrase = library.retrieve("舌簧", station="WS1001")
            self.assertTrue(paraphrase)
            self.assertTrue(all(item.metadata["station"] == "WS1001" for item in paraphrase))
            self.assertTrue(all(item.metadata["retrieved_by"] == ["semantic"] for item in paraphrase))
            library.ingest("manual.pdf", make_pdf("确认支架完整"), "WS1001")
            reopened = InstructionLibrary(path)
            reopened.embeddings = TestEmbeddings()
            self.assertEqual(reopened.vector_status(), status)
            self.assertTrue(all("卡扣" not in item.page_content for item in reopened.retrieve("卡扣", station="WS1001")))
