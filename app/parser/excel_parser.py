from io import BytesIO
from pathlib import Path
import re
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, range_boundaries
from openpyxl.utils.exceptions import InvalidFileException

from .base import BaseParser, ParsedDocument, ParsedElement


class ExcelParser(BaseParser):
    """解析 .xlsx/.xlsm 规则明细表，每个 Sheet 默认视为一张表。

    自动跳过顶部通栏合并标题，并通过合并范围识别多级表头。
    复杂布局可指定一基 header_row 和表头行数 header_rows（对所有 Sheet 生效）。
    不支持旧 .xls；公式只读取缓存值，不执行计算。
    """

    def __init__(
        self,
        header_row: int | None = None,
        header_rows: int | None = None,
        max_cells: int = 250_000,
        max_file_bytes: int = 20 * 1024 * 1024,
    ):
        if header_row is not None and header_row < 1:
            raise ValueError("header_row 必须从 1 开始")
        if header_rows is not None and header_rows < 1:
            raise ValueError("header_rows 必须大于 0")
        if max_cells < 1 or max_file_bytes < 1:
            raise ValueError("读取规模限制必须大于 0")
        self.header_row = header_row
        self.header_rows = header_rows
        self.max_cells = max_cells
        self.max_file_bytes = max_file_bytes

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if path.suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("仅支持 .xlsx/.xlsm，请先转换旧版 .xls 文件")
        if path.stat().st_size > self.max_file_bytes:
            raise ValueError("Excel 文件过大，请拆分后再解析")
        # 普通读取模式才能获取 merged_cells；全程不写入或保存工作簿。
        workbook = load_workbook(path, read_only=False, data_only=True)
        elements: list[ParsedElement] = []

        try:
            for sheet in workbook.worksheets:
                elements.extend(self._parse_sheet(sheet))
        finally:
            workbook.close()

        return self._build_doc(str(path), elements)

    def extract_instruction_rows(self, content: bytes) -> list[ParsedElement]:
        """保留任意版式中的非空行和原始坐标，包括标题及说明行。"""
        if len(content) > self.max_file_bytes:
            raise ValueError("Excel 文件过大，请拆分后再解析。")
        try:
            workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
            try:
                elements = []
                remaining = self.max_cells
                for sheet in workbook.worksheets:
                    if sheet.sheet_state != "visible":
                        continue
                    # 忽略维度中的格式化空白列，仍用实际单元格坐标定位。
                    sheet.reset_dimensions()
                    sheet_rows = []
                    for row_number, row in enumerate(sheet.values, 1):
                        cells = [(column, str(value).strip()) for column, value in enumerate(row, 1)
                                 if value is not None and str(value).strip()]
                        remaining -= max(1, len(cells))
                        if remaining < 0:
                            raise ValueError("Excel 工作区过大，请拆分后再解析。")
                        if not cells:
                            continue
                        first, last = (f"{get_column_letter(c)}{row_number}" for c in (cells[0][0], cells[-1][0]))
                        location = first if first == last else f"{first}:{last}"
                        sheet_rows.append(ParsedElement(
                            type="table",
                            text="\n".join(f"{get_column_letter(column)}{row_number} {value}" for column, value in cells),
                            metadata={"sheet": sheet.title, "cell_range": location},
                        ))
                    elements.extend(self._combine_step_rows(sheet_rows))
                return elements
            finally:
                workbook.close()
        except (BadZipFile, InvalidFileException, KeyError, OSError, ParseError, SyntaxError) as exc:
            raise ValueError("Excel 文件无法读取，请提供有效的 .xlsx 或 .xlsm 文件。") from exc

    @staticmethod
    def _combine_step_rows(rows: list[ParsedElement]) -> list[ParsedElement]:
        if not any("作业分解指导书" in item.text for item in rows[:5]):
            return rows
        steps: list[list[ParsedElement]] = []
        for item in rows:
            column, row, _, _ = range_boundaries(item.metadata["cell_range"])
            previous = steps[-1] if steps else []
            if column > 1 and previous and re.match(r"A\d+ \d+[.．、)]", previous[0].text):
                _, previous_row, _, _ = range_boundaries(previous[-1].metadata["cell_range"])
                if row - previous_row <= 2:
                    previous.append(item)
                    continue
            steps.append([item])
        results = []
        for step in steps:
            if len(step) == 1:
                results.append(step[0])
                continue
            bounds = [range_boundaries(item.metadata["cell_range"]) for item in step]
            last_column = get_column_letter(max(bound[2] for bound in bounds))
            results.append(ParsedElement(
                type="table",
                text="\n".join(item.text for item in step),
                metadata={"sheet": step[0].metadata["sheet"],
                          "cell_range": f"A{bounds[0][1]}:{last_column}{bounds[-1][3]}"},
            ))
        return results

    def _parse_sheet(self, sheet) -> list[ParsedElement]:
        if sheet.max_row * sheet.max_column > self.max_cells:
            raise ValueError(f"Sheet {sheet.title!r} 的区域过大，请清理空白格式或拆分")

        values = {}
        for row in sheet.iter_rows():
            for cell in row:
                text = "" if cell.value is None else str(cell.value).strip()
                if text:
                    values[(cell.row, cell.column)] = text
        if not values:
            return []

        merges = list(sheet.merged_cells.ranges)
        # 只填补真正的合并区域，绝不对普通空值做前向填充。
        for merged in merges:
            value = values.get((merged.min_row, merged.min_col), "")
            if value:
                for row in range(merged.min_row, merged.max_row + 1):
                    for col in range(merged.min_col, merged.max_col + 1):
                        values[(row, col)] = value

        first_col = min(col for _, col in values)
        last_col = max(col for _, col in values)
        last_row = max(row for row, _ in values)
        header_start = self.header_row or min(row for row, _ in values)
        titles = []
        if self.header_row is None:
            while header_start <= last_row:
                if not any(values.get((header_start, c)) for c in range(first_col, last_col + 1)):
                    header_start += 1
                    continue
                title_merge = next(
                    (m for m in merges if m.min_row == header_start
                     and m.min_col == first_col and m.max_col == last_col
                     and last_col > first_col),
                    None,
                )
                if title_merge is None:
                    break
                titles.append(values.get((header_start, first_col), ""))
                header_start = title_merge.max_row + 1

        if header_start > last_row:
            # 只有标题时仍保留文本，不误当成数据记录。
            return [ParsedElement(type="title", text=t, level=1,
                                  metadata={"sheet": sheet.title, "format": "excel"})
                    for t in titles]

        header_end = header_start + (self.header_rows or 1) - 1
        if self.header_rows is None:
            # 横向分组表头下面还有一层；纵向合并表头覆盖到其末行。
            for row in range(header_start, last_row + 1):
                if row > header_end:
                    break
                for merged in merges:
                    if merged.min_row == row:
                        bottom = merged.max_row + (merged.max_col > merged.min_col)
                        header_end = max(header_end, bottom)
        if header_end > last_row:
            raise ValueError(f"Sheet {sheet.title!r} 的表头超出有效区域，请指定正确表头")

        headers = []
        used = set()
        for col in range(first_col, last_col + 1):
            parts = []
            for row in range(header_start, header_end + 1):
                value = values.get((row, col), "")
                if value and (not parts or parts[-1] != value):
                    parts.append(value)
            label = " / ".join(parts) or f"列{get_column_letter(col)}"
            unique_label = label
            suffix = 2
            while unique_label in used:
                unique_label = f"{label} ({suffix})"
                suffix += 1
            used.add(unique_label)
            headers.append(unique_label)

        elements = []
        title = " > ".join(titles)
        for row in range(header_end + 1, last_row + 1):
            row_values = [values.get((row, col), "") for col in range(first_col, last_col + 1)]
            if not any(row_values):
                continue
            fields = "；".join(f"{key}：{value}" for key, value in zip(headers, row_values))
            prefix = f"工作表：{sheet.title}；行号：{row}"
            if title:
                prefix += f"；标题：{title}"
            elements.append(ParsedElement(
                type="table",
                text=f"{prefix}\n{fields}",
                metadata={
                    "sheet": sheet.title,
                    "format": "excel",
                    "row_number": row,
                    "cell_range": f"{get_column_letter(first_col)}{row}:{get_column_letter(last_col)}{row}",
                    "header_range": f"{get_column_letter(first_col)}{header_start}:{get_column_letter(last_col)}{header_end}",
                    "headers": headers.copy(),
                    "title": title,
                    "merged_ranges": [str(m) for m in merges if m.min_row <= row <= m.max_row],
                },
            ))
        return elements


class EXCELParser(ExcelParser):
    """兼容旧类名；调用方式仍为 EXCELParser().parse(file_path)。"""

    pass
