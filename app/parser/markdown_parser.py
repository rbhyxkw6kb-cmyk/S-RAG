import re
from typing import List

from .base import BaseParser, ParsedDocument, ParsedElement


class MarkdownParser(BaseParser):
    """Markdown 解析器，保留标题层级、代码块、表格"""

    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    CODE_PATTERN = re.compile(r'^```(\w*)$')
    TABLE_ROW_PATTERN = re.compile(r'^\|.*\|$')

    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        elements: List[ParsedElement] = []
        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n")

            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                elements.append(ParsedElement(type="title", text=text, level=level))
                i += 1
                continue

            code_match = self.CODE_PATTERN.match(line)
            if code_match:
                lang = code_match.group(1)
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i].rstrip("\n"))
                    i += 1
                elements.append(
                    ParsedElement(
                        type="code",
                        text="\n".join(code_lines),
                        metadata={"language": lang},
                    )
                )
                i += 1
                continue

            if self.TABLE_ROW_PATTERN.match(line):
                table_lines = []
                while i < len(lines) and self.TABLE_ROW_PATTERN.match(lines[i].rstrip("\n")):
                    table_lines.append(lines[i].rstrip("\n"))
                    i += 1
                elements.append(ParsedElement(type="table", text="\n".join(table_lines)))
                continue

            if line.strip():
                para_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not self._is_block_start(lines[i]):
                    para_lines.append(lines[i].rstrip("\n"))
                    i += 1
                elements.append(ParsedElement(type="paragraph", text=" ".join(para_lines)))
                continue

            i += 1

        return self._build_doc(file_path, elements)

    def _is_block_start(self, line: str) -> bool:
        return bool(
            self.HEADING_PATTERN.match(line)
            or self.CODE_PATTERN.match(line)
            or self.TABLE_ROW_PATTERN.match(line)
        )