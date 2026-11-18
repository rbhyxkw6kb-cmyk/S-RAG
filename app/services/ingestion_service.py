import asyncio
from typing import List

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.chunker.semantic_chunker import SemanticChunker
from app.core.config import settings
from app.parser.base import BaseParser
from app.retriever.bm25_retriever import BM25Retriever
from app.retriever.vector_retriever import VectorRetriever


class IngestionService:
    """文档入库服务：解析 → 分块 → 向量化 → 写入 Milvus + Whoosh"""

    def __init__(self):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.embedding_batch_size,
            },
            query_instruction=settings.query_instruction,
        )
        self.chunker = SemanticChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._init_milvus()
        self.bm25 = BM25Retriever(settings.whoosh_index_dir)
        self.vector_retriever = VectorRetriever(self.collection, self.embeddings)

    def _init_milvus(self):
        connections.connect(
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        if not utility.has_collection(settings.milvus_collection):
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="知源知识库文本块")
            self.collection = Collection(settings.milvus_collection, schema)
            index_params = {
                "metric_type": settings.milvus_metric_type,
                "index_type": settings.milvus_index_type,
                "params": {"M": 16, "efConstruction": 256},
            }
            self.collection.create_index("embedding", index_params)
        else:
            self.collection = Collection(settings.milvus_collection)
        self.collection.load()

    async def ingest_file(self, file_path: str, parser: BaseParser) -> dict:
        """入库单个文档"""
        parsed_doc = parser.parse(file_path)
        chunks = self.chunker.split(parsed_doc)
        if not chunks:
            return {"doc_id": parsed_doc.doc_id, "chunks": 0, "status": "empty"}

        texts = [c.page_content for c in chunks]
        vectors = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.embeddings.embed_documents(texts),
        )

        entities = [
            [c.metadata["chunk_id"] for c in chunks],
            [c.metadata["doc_id"] for c in chunks],
            [c.metadata["source"] for c in chunks],
            [c.metadata["section"] for c in chunks],
            [c.page_content for c in chunks],
            vectors,
        ]
        self.collection.insert(entities)
        self.collection.flush()
        return {
            "doc_id": parsed_doc.doc_id,
            "source": parsed_doc.source,
            "chunks": len(chunks),
            "total_chars": parsed_doc.total_chars,
            "status": "ready",
        }