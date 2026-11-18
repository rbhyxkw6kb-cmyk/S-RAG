from threading import Lock

from app.core.config import settings


class LocalEmbeddings:
    """FastEmbed 的本地 ONNX 模型，文档和问题使用同一向量空间。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_id = f"fastembed-0.7.4:{model_name}:{settings.rag_embedding_revision}"
        self._model = None
        self._lock = Lock()

    def _runtime(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=self.model_name,
                        cache_dir=settings.rag_embedding_cache,
                        threads=settings.rag_embedding_threads,
                        specific_model_path=settings.rag_embedding_path,
                    )
        return self._model

    def _normalized(self, vectors, expected: int):
        import numpy as np

        vectors = np.asarray(list(vectors), dtype="<f4")
        if vectors.ndim != 2 or len(vectors) != expected or not np.isfinite(vectors).all():
            raise ValueError("嵌入模型返回了无效向量。")
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norm <= 0).any():
            raise ValueError("嵌入模型返回了零向量。")
        return vectors / norm

    def documents(self, texts: list[str]) -> list[bytes]:
        vectors = self._normalized(self._runtime().passage_embed(texts, batch_size=16), len(texts))
        return [vector.astype("<f4", copy=False).tobytes() for vector in vectors]

    def query(self, text: str):
        return self._normalized(self._runtime().query_embed(text), 1)[0]

    @staticmethod
    def rank(query, vectors: list[tuple[int, bytes]], floor: float, limit: int) -> list[int]:
        import numpy as np

        if not vectors:
            return []
        row_ids, blobs = zip(*vectors)
        scores = np.frombuffer(b"".join(blobs), dtype="<f4").reshape(-1, len(query)) @ query
        positions = np.argsort(-scores, kind="stable")[:limit]
        return [row_ids[i] for i in positions if scores[i] >= floor]


class LocalReranker:
    def __init__(self):
        self._model = None
        self._lock = Lock()

    def rank(self, query: str, texts: list[str]) -> list[int]:
        import math

        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(
                        model_name=settings.rag_reranker_model,
                        cache_dir=settings.rag_embedding_cache,
                        threads=settings.rag_embedding_threads,
                        specific_model_path=settings.rag_reranker_path,
                    )
        scores = list(self._model.rerank(query, texts, batch_size=8))
        if len(scores) != len(texts) or not all(math.isfinite(score) for score in scores):
            raise ValueError("重排模型返回无效分数。")
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
