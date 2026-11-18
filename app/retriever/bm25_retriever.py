import os
from typing import List

import jieba
from langchain.schema import Document
from whoosh.analysis import Token, Tokenizer
from whoosh.fields import ID, STORED, Schema, TEXT
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser


class ChineseTokenizer(Tokenizer):
    """Whoosh 中文分词器，基于 jieba"""

    def __call__(
        self,
        value,
        positions=False,
        chars=False,
        keeporiginal=False,
        removestops=True,
        start_pos=0,
        start_char=0,
        mode="",
        **kwargs,
    ):
        token = Token(positions, chars, removestops=removestops, mode=mode, **kwargs)
        for word in jieba.cut(value):
            if word.strip():
                token.text = word
                token.boost = 1.0
                token.pos = start_pos
                token.startchar = start_char
                token.endchar = start_char + len(word)
                yield token
                start_pos += 1
                start_char += len(word)


class BM25Retriever:
    """基于 Whoosh 的 BM25 检索器"""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.schema = Schema(
            chunk_id=ID(stored=True, unique=True),
            doc_id=ID(stored=True),
            source=STORED(),
            section=STORED(),
            text=TEXT(analyzer=ChineseTokenizer(), stored=True),
        )
        if os.listdir(index_dir):
            self.ix = open_dir(index_dir)
        else:
            self.ix = create_in(index_dir, self.schema)

    def add_documents(self, chunks: List[Document]):
        """批量写入索引"""
        writer = self.ix.writer()
        for chunk in chunks:
            writer.add_document(
                chunk_id=chunk.metadata["chunk_id"],
                doc_id=chunk.metadata["doc_id"],
                source=chunk.metadata["source"],
                section=chunk.metadata["section"],
                text=chunk.page_content,
            )
        writer.commit()

    def retrieve(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        BM25 检索，返回 [(Document, score, rank), ...]
        rank 从 1 开始，用于 RRF 融合
        """
        results = []
        with self.ix.searcher() as searcher:
            query_obj = QueryParser("text", self.schema).parse(query)
            hits = searcher.search(query_obj, limit=top_k)
            for rank, hit in enumerate(hits, 1):
                doc = Document(
                    page_content=hit["text"],
                    metadata={
                        "chunk_id": hit["chunk_id"],
                        "doc_id": hit["doc_id"],
                        "source": hit["source"],
                        "section": hit["section"],
                    },
                )
                results.append((doc, hit.score, rank))
        return results