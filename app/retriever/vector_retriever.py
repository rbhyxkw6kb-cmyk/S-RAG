from typing import List
from langchain.schema import Document
from pymilvus import Collection
from app.core.config import settings

class VectorRetriever:
    """基于 Milvus 的向量检索器"""

    def __init__(self, collection: Collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.search_params = {"metric_type": "IP", "params": {"ef": 128}}

    def retrieve(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        向量检索，返回 [(Document, score, rank), ...]
        """
        query_with_instruction = f"{settings.query_instruction}{query}"
        query_vector = self.embedding_model.embed_query(query_with_instruction)

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "source", "section", "text"],
        )

        retrieved = []
        for rank, hit in enumerate(results[0], 1):
            doc = Document(
                page_content=hit.entity.get("text"),
                metadata={
                    "chunk_id": hit.entity.get("chunk_id"),
                    "doc_id": hit.entity.get("doc_id"),
                    "source": hit.entity.get("source"),
                    "section": hit.entity.get("section"),
                },
            )
            retrieved.append((doc, hit.score, rank))
        return retrieved