from threading import Lock
from typing import List

import pymupdf as fitz

from .base import BaseParser, ParsedDocument, ParsedElement

_pdf_lock = Lock()

class PDFParser(BaseParser):
    """PDF 文档解析器，基于 PyMuPDF，保留标题层级和布局信息"""

    FONT_SIZE_THRESHOLDS = {16: 1, 14: 2, 12: 3, 11: 4}

    @staticmethod
    def extract_pages(content: bytes) -> list[str]:
        """按物理页提取可搜索文本，保留空页的位置。"""
        try:
            with _pdf_lock, fitz.open(stream=content, filetype="pdf") as doc:
                if not doc.is_pdf or doc.needs_pass:
                    raise ValueError("请提供未加密的 PDF 文件。")
                return [page.get_text("text", sort=True).strip() for page in doc]
        except (fitz.FileDataError, fitz.EmptyFileError) as exc:
            raise ValueError("PDF 文件无法读取。") from exc

    def parse(self, file_path: str) -> ParsedDocument:
        doc = fitz.open(file_path)
        elements: List[ParsedElement] = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text:
                        continue

                    max_font_size = max(span["size"] for span in line["spans"])
                    level = self._detect_heading_level(max_font_size)

                    elem_type = "title" if level > 0 else "paragraph"
                    elements.append(
                        ParsedElement(
                            type=elem_type,
                            text=text,
                            level=level,
                            metadata={"page": page_num + 1},
                        )
                    )

        doc.close()
        return self._build_doc(file_path, elements)

    def _detect_heading_level(self, font_size: float) -> int:
        for threshold, level in sorted(self.FONT_SIZE_THRESHOLDS.items(), reverse=True):
            if font_size >= threshold:
                return level
        return 0
