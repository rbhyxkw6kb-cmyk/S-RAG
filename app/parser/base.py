from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import hashlib

@dataclass
class ParsedElement:
    """解析后的结构化元素"""
    type: str  # title / paragraph / table / code / list
    text: str
    level: int = 0  # 标题层级，1-6
    metadata: dict = field(default_factory=dict)

@dataclass
class ParsedDocument:
    """解析后的完整文档"""
    doc_id: str
    source: str  # 文件名
    elements: List[ParsedElement]
    total_chars: int = 0

    @staticmethod
    def generate_doc_id(file_path: str) -> str:
        return hashlib.md5(file_path.encode()).hexdigest()[:16]

class BaseParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文档，返回结构化元素列表"""
        pass

    def _build_doc(self, file_path: str, elements: List[ParsedElement]) -> ParsedDocument:
        total = sum(len(e.text) for e in elements)
        return ParsedDocument(
            doc_id=ParsedDocument.generate_doc_id(file_path),
            source=file_path.split("/")[-1],
            elements=elements,
            total_chars=total,
        )