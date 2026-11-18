from __future__ import annotations
import re
from typing import Dict, List, TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.documents import Document


def parse_citations(answer: str, context_docs: List[Document]) -> List[Dict]:
    """
    解析回答中的引用编号 [n]，映射到具体的来源元数据
    返回去重后的引用列表，按出现顺序排列
    """
    pattern = re.compile(r'\[(\d+(?:[,\-]\d+)*)\]')
    cited_indices = set()

    for match in pattern.finditer(answer):
        raw = match.group(1)
        for part in raw.split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                cited_indices.update(range(max(1, int(start)), min(len(context_docs), int(end)) + 1))
            else:
                cited_indices.add(int(part))

    valid_indices = [i for i in sorted(cited_indices) if 1 <= i <= len(context_docs)]

    citations = []
    for idx in valid_indices:
        doc = context_docs[idx - 1]
        citations.append(
            {
                "index": idx,
                "source": doc.metadata.get("source", "未知"),
                "section": doc.metadata.get("section", "正文"),
                "page": doc.metadata.get("page"),
                "sheet": doc.metadata.get("sheet"),
                "cell_range": doc.metadata.get("cell_range"),
                "file_url": doc.metadata.get("file_url"),
                "station": doc.metadata.get("station"),
                "pdf_url": doc.metadata.get("pdf_url"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "rerank_score": doc.metadata.get("rerank_score", 0),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            }
        )
    return citations
