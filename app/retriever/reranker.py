from typing import List

from FlagEmbedding import FlagReranker
from langchain.schema import Document

from app.core.config import settings


class BgeReranker:
    """
    bge-reranker 重排序器
    Cross-Encoder 架构：将 query 和 doc 拼接后输入模型，输出相关性得分
    """

    def __init__(self):
        self.model = FlagReranker(
            settings.reranker_model,
            use_fp16=True,
            device=settings.reranker_device,
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Document]:
        """
        对候选文档重排序
        Args:
            query: 用户查询
            documents: 混合检索返回的候选列表
            top_k: 返回前 K 个
            threshold: 最低得分阈值，低于此值的文档被过滤
        Returns:
            重排序后的文档列表，metadata 中含 rerank_score
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        scored_docs = []
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)
            scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        filtered = [d for d in scored_docs if d.metadata["rerank_score"] >= threshold]
        return filtered[:top_k]