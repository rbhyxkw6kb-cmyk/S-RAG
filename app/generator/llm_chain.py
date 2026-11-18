import re
from typing import AsyncIterator, List
from langchain.llms import VLLMOpenAI
from langchain.schema import Document
from app.core.config import settings
from .prompt_builder import SYSTEM_PROMPT, build_chat_prompt
from .citation_parser import parse_citations

class AnswerGenerator:
    """
    回答生成器：组装 Prompt → 调用 vLLM → 流式输出  解析引用
    """

    def __init__(self):
        self.llm = VLLMOpenAI(
            openai_api_key=settings.llm_api_key,
            openai_api_base=settings.llm_base_url,
            model_name=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            streaming=True,
        )

    async def generate_stream(
        self,
        query: str,
        context_docs: List[Document],
        chat_history: str = "",
    ) -> AsyncIterator[dict]:
        """流式生成回答，yield 每个 token，最后 yield 一个包含 citations 的结束消息"""
        user_prompt = build_chat_prompt(query, context_docs, chat_history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        full_answer = ""
        async for chunk in self.llm.astream(messages):
            token = chunk if isinstance(chunk, str) else chunk.content
            full_answer += token
            yield {"type": "token", "content": token}

        citations = parse_citations(full_answer, context_docs)
        yield {
            "type": "done",
            "answer": full_answer,
            "citations": citations,
            "context_count": len(context_docs),
        }