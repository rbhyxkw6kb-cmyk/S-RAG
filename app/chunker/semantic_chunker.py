from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.parser.base import ParsedDocument, ParsedElement


class SemanticChunker:
    """
    语义感知分块器：
    1. 优先按标题层级切分（每个标题下的内容为一个块）
    2. 超过 chunk_size 的段落再用递归字符分割补切
    3. 每个块携带完整的章节路径元数据（如 "第三章 > 3.1 架构设计"）
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ";", "，", " ", ""],
        )

    def split(self, parsed_doc: ParsedDocument) -> List[Document]:
        documents: List[Document] = []
        section_stack: List[str] = []
        current_buffer: List[str] = []

        def flush_buffer():
            if not current_buffer:
                return
            text = "\n".join(current_buffer)
            section_path = " > ".join(section_stack) if section_stack else "正文"
            if len(text) > self.chunk_size * 1.5:
                sub_chunks = self._fallback_splitter.split_text(text)
                for idx, chunk in enumerate(sub_chunks):
                    documents.append(self._make_doc(parsed_doc, chunk, section_path, idx))
            else:
                documents.append(self._make_doc(parsed_doc, text, section_path, 0))
            current_buffer.clear()

        for elem in parsed_doc.elements:
            if elem.type == "title":
                flush_buffer()
                while section_stack and len(section_stack) >= elem.level:
                    section_stack.pop()
                section_stack.append(elem.text)
                current_buffer.append(f"【{elem.text}】")
            else:
                current_buffer.append(elem.text)

        flush_buffer()
        return documents

    def _make_doc(
        self,
        parsed_doc: ParsedDocument,
        text: str,
        section_path: str,
        sub_idx: int,
    ) -> Document:
        return Document(
            page_content=text,
            metadata={
                "doc_id": parsed_doc.doc_id,
                "source": parsed_doc.source,
                "section": section_path,
                "chunk_id": f"{parsed_doc.doc_id}_{len(text)}_{sub_idx}",
                "char_count": len(text),
            },
        )