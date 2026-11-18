"""Extract workbook pictures at their original worksheet anchors."""

import logging
import posixpath
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter, range_boundaries
from PIL import Image

logger = logging.getLogger(__name__)
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
HEADER = re.compile(r"^([A-Z]+\d+)\s+(?:图例|圖例|illustrations?)(?:\s|$)", re.I | re.M)
NUMBERED = re.compile(r"^A\d+\s+\d+\s*[.．、)）]")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@dataclass
class Figure:
    sheet: str
    first_row: int
    last_row: int
    anchor: str
    png: bytes


def relationships(archive: ZipFile, part: str) -> dict:
    parent, name = posixpath.split(part)
    path = posixpath.join(parent, "_rels", f"{name}.rels")
    if path not in archive.namelist():
        return {}
    result = {}
    for rel in ElementTree.fromstring(archive.read(path)):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target", "")
        local = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(parent, target))
        if local.startswith("xl/"):
            result[rel.get("Id")] = local
    return result


def drawing_pictures(archive: ZipFile, part: str):
    links = relationships(archive, part)
    for anchor in ElementTree.fromstring(archive.read(part)):
        marker = anchor.find("./{*}from")
        if marker is None:
            continue
        row = int(marker.findtext("{*}row", "-1")) + 1
        column = int(marker.findtext("{*}col", "-1")) + 1
        if not 1 <= row <= 1048576 or not 1 <= column <= 16384:
            continue
        for picture in anchor.findall("./{*}pic"):
            blip = picture.find("./{*}blipFill/{*}blip")
            path = links.get(blip.get(REL + "embed")) if blip is not None else None
            if path and PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS:
                yield row, column, path, picture


def prepare_picture(archive: ZipFile, path: str, picture):
    if archive.getinfo(path).file_size > 32 * 1024**2:
        return None
    with Image.open(BytesIO(archive.read(path))) as original:
        if original.width * original.height > 32_000_000:
            return None
        crop = picture.find("./{*}blipFill/{*}srcRect")
        values = [max(0, int(crop.get(key, 0))) / 100000 for key in ("l", "t", "r", "b")] if crop is not None else [0] * 4
        left, top, right, bottom = values
        if left + right >= 1 or top + bottom >= 1:
            return None
        image = original.crop((round(original.width * left), round(original.height * top),
                               round(original.width * (1 - right)), round(original.height * (1 - bottom))))
        if not image.width or not image.height:
            return None
        image = image.convert("RGBA")
        image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    transform = picture.find("./{*}spPr/{*}xfrm")
    box = None
    if transform is not None:
        origin = transform.find("{*}off")
        size = transform.find("{*}ext")
        if origin is not None and size is not None:
            x, y, width, height = map(int, (origin.get("x", 0), origin.get("y", 0),
                                           size.get("cx", 0), size.get("cy", 0)))
            if width > 0 and height > 0:
                box = x, y, x + width, y + height
        if transform.get("flipH") in {"1", "true"}:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if transform.get("flipV") in {"1", "true"}:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotation = int(transform.get("rot", 0)) / 60000
        if rotation:
            image = image.rotate(-rotation, expand=True)
            box = None
    return image, box


def overlap(first, second) -> bool:
    if first is None or second is None:
        return False
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    shared = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    return shared > 0.5 * min((x2 - x1) * (y2 - y1), (a2 - a1) * (b2 - b1))


def render_group(pictures) -> bytes:
    canvas = pictures[0][0]
    if len(pictures) > 1:
        left = min(box[0] for _, box in pictures)
        top = min(box[1] for _, box in pictures)
        width = max(box[2] for _, box in pictures) - left
        height = max(box[3] for _, box in pictures) - top
        ratio = 1400 / max(width, height)
        canvas = Image.new("RGBA", (max(1, round(width * ratio)), max(1, round(height * ratio))))
        # Draw in Excel's order, preserving transparent raster overlays.
        for image, (x1, y1, x2, y2) in pictures:
            image = image.resize((max(1, round((x2 - x1) * ratio)), max(1, round((y2 - y1) * ratio))),
                                 Image.Resampling.LANCZOS)
            canvas.alpha_composite(image, (round((x1 - left) * ratio), round((y1 - top) * ratio)))
    canvas.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def extract_excel_figures(content: bytes, elements) -> list[Figure]:
    sheets = {}
    for item in elements:
        sheets.setdefault(item.metadata["sheet"], []).append(item)

    figures = []
    with ZipFile(BytesIO(content)) as archive:
        workbook_path = "xl/workbook.xml"
        workbook = ElementTree.fromstring(archive.read(workbook_path))
        worksheets = relationships(archive, workbook_path)
        for node in workbook.findall("./{*}sheets/{*}sheet"):
            name = node.get("name")
            rows = sheets.get(name)
            sheet_path = worksheets.get(node.get(REL + "id"))
            if not rows or not sheet_path:
                continue
            root = ElementTree.fromstring(archive.read(sheet_path))
            sheet_links = relationships(archive, sheet_path)
            merges = [range_boundaries(cell.get("ref")) for cell in root.findall("./{*}mergeCells/{*}mergeCell")]
            merge_ends = {(r1, c1): c2 for c1, r1, c2, _ in merges}
            headers = [coordinate_to_tuple(match[1]) for item in rows for match in HEADER.finditer(item.text)]
            row_bounds = [(range_boundaries(item.metadata["cell_range"]), item) for item in rows]
            blocks = {}
            for relation in root.findall("./{*}drawing"):
                drawing = sheet_links.get(relation.get(REL + "id"))
                if not drawing:
                    continue
                for row, column, path, picture in drawing_pictures(archive, drawing):
                    for (_, first, _, last), item in row_bounds:
                        if not first <= row <= last:
                            continue
                        if headers:
                            if not any(first > head and start <= column <= merge_ends.get((head, start), start)
                                       for head, start in headers):
                                continue
                        elif not NUMBERED.match(item.text) or column == 1:
                            continue
                        try:
                            prepared = prepare_picture(archive, path, picture)
                        except (OSError, ValueError, KeyError, Image.DecompressionBombError):
                            logger.warning("Unable to render embedded Excel picture")
                            continue
                        if prepared:
                            blocks.setdefault((first, last), []).append((row, column, prepared))
                        break
            for (first, last), pictures in blocks.items():
                groups = []
                for row, column, prepared in pictures:
                    for _, _, group in groups:
                        if any(overlap(box, prepared[1]) for _, box in group):
                            group.append(prepared)
                            break
                    else:
                        groups.append((row, column, [prepared]))
                for row, column, group in groups[:8]:
                    figures.append(Figure(name, first, last, f"{get_column_letter(column)}{row}", render_group(group)))
    return figures
