"""Excel parsing tests; no model, database or GPU required."""

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from app.parser.excel_parser import EXCELParser, ExcelParser


class ExcelParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.xlsx"
        self.book = Workbook()
        self.addCleanup(self.book.close)
        self.sheet = self.book.active
        self.sheet.title = "Staff"

    def parse(self, **kwargs):
        self.book.save(self.path)
        original = self.path.read_bytes()
        doc = ExcelParser(**kwargs).parse(str(self.path))
        self.assertEqual(original, self.path.read_bytes())
        self.assertEqual(doc.total_chars, sum(len(e.text) for e in doc.elements))
        return doc

    def test_records_preserve_zero_false_and_empty_values(self):
        self.sheet.append(["Department", "Name", "Count", "Active"])
        self.sheet.append(["R&D", "Alice", 0, False])
        self.sheet.append([None, "Bob", None, True])
        doc = self.parse()
        self.assertEqual(len(doc.elements), 2)
        self.assertIn("Count：0", doc.elements[0].text)
        self.assertIn("Active：False", doc.elements[0].text)
        self.assertIn("Department：；", doc.elements[1].text)
        self.assertNotIn("R&D", doc.elements[1].text)
        self.assertEqual(doc.elements[1].metadata["cell_range"], "A3:D3")

    def test_title_multilevel_headers_and_vertical_merge(self):
        self.sheet.append(["Annual report", None, None])
        self.sheet.merge_cells("A1:C1")
        self.sheet.append(["Department", "2025", None])
        self.sheet.append([None, "Income", "Cost"])
        self.sheet.merge_cells("A2:A3")
        self.sheet.merge_cells("B2:C2")
        self.sheet.append(["R&D", 100, 50])
        self.sheet.append([None, 200, 80])
        self.sheet.merge_cells("A4:A5")
        doc = self.parse()
        self.assertEqual(len(doc.elements), 2)
        elem = doc.elements[1]
        self.assertIn("Department：R&D", elem.text)
        self.assertIn("2025 / Income：200", elem.text)
        self.assertEqual(elem.metadata["title"], "Annual report")
        self.assertEqual(elem.metadata["header_range"], "A2:C3")
        self.assertEqual(elem.metadata["row_number"], 5)
        self.assertEqual(elem.metadata["merged_ranges"], ["A4:A5"])

    def test_horizontal_data_merge(self):
        self.sheet.append(["First", "Second", "Third"])
        self.sheet.append(["Shared", None, "Other"])
        self.sheet.merge_cells("A2:B2")
        elem = self.parse().elements[0]
        self.assertIn("First：Shared；Second：Shared", elem.text)

    def test_manual_header_location_and_depth(self):
        self.sheet.append(["Notes"])
        self.sheet.append(["Group", "Group"])
        self.sheet.append(["Name", "Value"])
        self.sheet.append(["Alice", 42])
        elem = self.parse(header_row=2, header_rows=2).elements[0]
        self.assertEqual(elem.metadata["headers"], ["Group / Name", "Group / Value"])
        self.assertEqual(elem.metadata["row_number"], 4)

    def test_full_width_group_can_be_selected_as_header(self):
        self.sheet.append(["2025", None])
        self.sheet.merge_cells("A1:B1")
        self.sheet.append(["Income", "Cost"])
        self.sheet.append([100, 50])
        elem = self.parse(header_row=1, header_rows=2).elements[0]
        self.assertEqual(elem.metadata["headers"], ["2025 / Income", "2025 / Cost"])

    def test_duplicate_and_missing_headers(self):
        self.sheet.append(["Name", "Name", None])
        self.sheet.append(["Alice", "Bob", "Other"])
        headers = self.parse().elements[0].metadata["headers"]
        self.assertEqual(headers, ["Name", "Name (2)", "列C"])

    def test_multiple_sheets_and_blank_rows(self):
        self.sheet.append(["Name"])
        self.sheet.append(["Alice"])
        self.sheet.append([None])
        self.sheet.append(["Bob"])
        other = self.book.create_sheet("Other")
        other.append(["Name"])
        other.append(["Carol"])
        self.book.create_sheet("Empty")
        doc = self.parse()
        self.assertEqual(len(doc.elements), 3)
        self.assertEqual(doc.elements[1].metadata["row_number"], 4)
        self.assertEqual(doc.elements[2].metadata["sheet"], "Other")
        self.assertEqual(len(EXCELParser().parse(str(self.path)).elements), 3)

    def test_empty_workbook(self):
        self.assertEqual(self.parse().elements, [])

    def test_formula_without_cache_is_not_calculated(self):
        self.sheet.append(["Name", "Value"])
        self.sheet.append(["Alice", "=1+1"])
        self.assertIn("Value：", self.parse().elements[0].text)
        self.assertNotIn("Value：2", self.parse().elements[0].text)

    def test_invalid_options_and_limits(self):
        with self.assertRaises(ValueError):
            ExcelParser(header_rows=0)
        with self.assertRaises(ValueError):
            ExcelParser(header_row=0)
        with self.assertRaises(ValueError):
            ExcelParser().parse("old.xls")
        self.sheet.append(["Name", "Value"])
        self.sheet.append(["Alice", 1])
        with self.assertRaises(ValueError):
            self.parse(max_cells=1)
        with self.assertRaises(ValueError):
            self.parse(max_file_bytes=1)
        with self.assertRaises(ValueError):
            self.parse(header_row=1, header_rows=10)


if __name__ == "__main__":
    unittest.main()

