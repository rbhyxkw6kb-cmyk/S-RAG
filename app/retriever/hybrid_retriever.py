from typing import Dict, List
from langchain.schema import Document
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever

class HybridRetriever:
    """
    混合检索器：BM25 + 向量并行检索，RRF 融合
    RRF 公式：score = Σ 1 / (k + rank_i)，默认 k=60
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = 60,
        fusion_top_k: int = 20,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k
        self.fusion_top_k = fusion_top_k

    def retrieve(self, query: str) -> List[Document]:
        """执行混合检索，返回融合后的 Top-K 文档列表"""
        bm25_results = self.bm25.retrieve(query, top_k=20)
        vector_results = self.vector.retrieve(query, top_k=20)

        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for doc, _, rank in bm25_results:
            cid = doc.metadata["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank)
            doc_map[cid] = doc

        for doc, _, rank in vector_results:
            cid = doc.metadata["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank)
            doc_map[cid] = doc

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_ids = sorted_ids[:self.fusion_top_k]

        results = []
        for cid in top_ids:
            doc = doc_map[cid]
            doc.metadata["rrf_score"] = round(fused_scores[cid], 6)
            results.append(doc)
        return results