import json
from typing import AsyncIterator, List

from langchain.schema import Document

from app.generator.llm_chain import AnswerGenerator
from app.memory.session_manager import SessionManager
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.reranker import BgeReranker


class ChatService:
    """问答编排服务：检索 → 重排序 → 生成 → 记忆"""

    def __init__(self, ingestion_service):
        self.ingestion = ingestion_service
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=ingestion_service.bm25,
            vector_retriever=ingestion_service.vector_retriever,
        )
        self.reranker = BgeReranker()
        self.generator = AnswerGenerator()
        self.session_mgr = SessionManager()

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """检索 + 重排序，返回最终上下文"""
        candidates = self.hybrid_retriever.retrieve(query)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked

    async def chat(self, query: str, session_id: str) -> dict:
        """非流式问答"""
        context_docs = self.retrieve(query)
        history = self.session_mgr.format_history_for_prompt(session_id)

        full_answer = ""
        citations = []
        async for msg in self.generator.generate_stream(query, context_docs, history):
            if msg["type"] == "token":
                full_answer += msg["content"]
            elif msg["type"] == "done":
                citations = msg["citations"]

        self.session_mgr.add_message(session_id, "user", query)
        self.session_mgr.add_message(session_id, "assistant", full_answer)
        return {"answer": full_answer, "citations": citations}

    async def chat_stream(self, query: str, session_id: str) -> AsyncIterator[str]:
        """流式问答，SSE 格式"""
        context_docs = self.retrieve(query)
        history = self.session_mgr.format_history_for_prompt(session_id)

        full_answer = ""
        async for msg in self.generator.generate_stream(query, context_docs, history):
            if msg["type"] == "token":
                full_answer += msg["content"]
                yield f"data: {json.dumps({'type': 'token', 'content': msg['content']}, ensure_ascii=False)}\n\n"
            elif msg["type"] == "done":
                self.session_mgr.add_message(session_id, "user", query)
                self.session_mgr.add_message(session_id, "assistant", full_answer)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"