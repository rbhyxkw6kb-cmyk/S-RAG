import hashlib
import logging
import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from time import time

from app.core.config import settings
from app.parser.excel_parser import ExcelParser
from app.parser.pdf_parser import PDFParser
from app.retriever.local_embeddings import LocalEmbeddings, LocalReranker

logger = logging.getLogger(__name__)

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
SUPPORTED_SUFFIXES = EXCEL_SUFFIXES | {".pdf"}
STATION_CODE = re.compile(r"(?<![a-z0-9])WS[0-9]{4}(?![a-z0-9])", re.IGNORECASE | re.ASCII)
EXCEL_STEP = re.compile(r"^A(?P<row>\d+)\s+(?P<number>\d+)\s*[.．、)）]\s*(?P<action>[^\n]+)")


def station_codes(text: str) -> set[str]:
    return {match.group().upper() for match in STATION_CODE.finditer(text)}


def file_limit_mb(name: str) -> int:
    return settings.max_pdf_mb if Path(name).suffix.lower() == ".pdf" else settings.max_excel_mb


@dataclass
class Passage:
    page_content: str
    metadata: dict


class EmbeddingUnavailable(RuntimeError):
    pass


def search_terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    terms = []
    for word in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", normalized):
        if "\u3400" <= word[0] <= "\u9fff" and len(word) > 1:
            terms.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            terms.append(word)
    return terms


def split_text(text: str):
    start = 0
    while start < len(text):
        end = min(start + 650, len(text))
        if end < len(text):
            line_end = text.rfind("\n", start + 325, end)
            if line_end != -1:
                end = line_end + 1
        yield text[start:end].strip()
        if end == len(text):
            break
        start = end - 90


class InstructionLibrary:
    def __init__(self, path: str):
        self.path = Path(path)
        self.embeddings = LocalEmbeddings(settings.rag_embedding_model) if settings.rag_embedding_model else None
        self.reranker = LocalReranker() if settings.rag_reranker_model else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS manuals (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, station TEXT NOT NULL,
                    pdf BLOB NOT NULL, pages INTEGER NOT NULL, text_pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL, active INTEGER NOT NULL,
                    index_version INTEGER NOT NULL DEFAULT 1
                );
                CREATE UNIQUE INDEX IF NOT EXISTS active_manual
                ON manuals(source, station) WHERE active = 1;
                CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                    doc_id UNINDEXED, page UNINDEXED, text UNINDEXED, terms, title_terms
                );
                CREATE TABLE IF NOT EXISTS spreadsheet_locations (
                    passage_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL,
                    sheet TEXT NOT NULL, cell_range TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS spreadsheet_document ON spreadsheet_locations(doc_id);
                CREATE TABLE IF NOT EXISTS passage_vectors (
                    passage_id INTEGER PRIMARY KEY, model_id TEXT NOT NULL, vector BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_steps (
                    session_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
                    passage_id INTEGER NOT NULL, sheet TEXT NOT NULL,
                    cell_range TEXT NOT NULL, expires_at INTEGER NOT NULL
                );
            """)
            columns = {column["name"] for column in db.execute("PRAGMA table_info(manuals)")}
            if "index_version" not in columns:
                db.execute("ALTER TABLE manuals ADD COLUMN index_version INTEGER NOT NULL DEFAULT 1")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def ingest(self, name: str, content: bytes, station: str = "") -> dict:
        name = PureWindowsPath(name).name
        station = station.strip()
        if not station:
            codes = station_codes(Path(name).stem)
            if len(codes) == 1:
                station = codes.pop()
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES or len(name) > 255:
            raise ValueError("请选择文件名不超过 255 个字符的 .xlsx、.xlsm 或 .pdf 文件。旧版 .xls 请另存为 .xlsx。")
        if len(content) > file_limit_mb(name) * 1024 ** 2:
            raise ValueError(f"文件大小上限为 {file_limit_mb(name)} MB。")
        if len(station) > 80:
            raise ValueError("工位名称最多 80 个字符。")
        doc_id = hashlib.sha256(name.encode() + b"\0" + station.encode() + b"\0" + content).hexdigest()
        index_version = 2 if suffix in EXCEL_SUFFIXES else 1
        with closing(self.connect()) as db:
            current = db.execute("SELECT active, index_version FROM manuals WHERE id = ?", (doc_id,)).fetchone()
        if current and current["active"] and current["index_version"] == index_version:
            return {"source": name, "status": "unchanged"}

        page_count = text_pages = sheet_count = 0
        if suffix == ".pdf":
            pages = PDFParser.extract_pages(content)
            page_count = len(pages)
            text_pages = sum(bool(page) for page in pages)
            if not text_pages:
                raise ValueError("PDF 没有可提取的文字，请先对扫描件做 OCR。")
            parts = [(page, page_text, "", "") for page, page_text in enumerate(pages, 1)]
        else:
            excel = ExcelParser(max_file_bytes=settings.max_excel_mb * 1024 ** 2)
            elements = excel.extract_instruction_rows(content)
            if not elements:
                raise ValueError("Excel 工作簿没有可检索的单元格内容。")
            parts = [(None, item.text, item.metadata["sheet"], item.metadata["cell_range"]) for item in elements]
            sheet_count = len({sheet for _, _, sheet, _ in parts})
        title_terms = " ".join(search_terms(f"{Path(name).stem} {station}"))
        rows = [
            (page, chunk, " ".join(search_terms(chunk)), sheet, cell_range)
            for page, source_text, sheet, cell_range in parts
            for chunk in split_text(source_text)
            if chunk
        ]
        vectors = self.encode([row[1] for row in rows]) if self.embeddings else None
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT id, index_version FROM manuals WHERE source = ? AND station = ? AND active = 1",
                (name, station),
            ).fetchone()
            if current and current["id"] == doc_id and current["index_version"] == index_version:
                return {"source": name, "status": "unchanged"}
            if current:
                db.execute("DELETE FROM passage_vectors WHERE passage_id IN "
                           "(SELECT rowid FROM passages WHERE doc_id = ?)", (current["id"],))
                db.execute("DELETE FROM spreadsheet_locations WHERE doc_id = ?", (current["id"],))
                db.execute("DELETE FROM passages WHERE doc_id = ?", (current["id"],))
                db.execute("UPDATE manuals SET active = 0 WHERE id = ?", (current["id"],))
            db.execute(
                "INSERT INTO manuals(id, source, station, pdf, pages, text_pages, chunks, active, index_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(id) DO UPDATE SET active = 1, chunks = excluded.chunks, "
                "index_version = excluded.index_version",
                (doc_id, name, station, content, page_count, text_pages, len(rows), index_version),
            )
            sheet_terms = {sheet: " ".join(search_terms(sheet)) for sheet in {row[3] for row in rows}}
            for index, (page, chunk, terms, sheet, cell_range) in enumerate(rows):
                cursor = db.execute("INSERT INTO passages VALUES (?, ?, ?, ?, ?)",
                                    (doc_id, page, chunk, terms, f"{title_terms} {sheet_terms[sheet]}"))
                if sheet:
                    db.execute("INSERT INTO spreadsheet_locations VALUES (?, ?, ?, ?)",
                               (cursor.lastrowid, doc_id, sheet, cell_range))
                if vectors is not None:
                    db.execute("INSERT INTO passage_vectors VALUES (?, ?, ?)",
                               (cursor.lastrowid, self.embeddings.model_id, vectors[index]))
        return {"source": name, "status": "ready", "pages": page_count,
                "text_pages": text_pages, "sheets": sheet_count}

    def scan(self) -> list[dict]:
        results = []
        paths = sorted(path for path in Path(settings.dataset_dir).rglob("*")
                       if path.is_file() and not path.name.startswith("~$")
                       and path.suffix.lower() in SUPPORTED_SUFFIXES)
        directory = Path(settings.dataset_dir)
        for path in paths:
            relative = path.relative_to(directory)
            station = relative.parent.as_posix() if len(relative.parts) > 1 else ""
            try:
                with path.open("rb") as file:
                    limit = file_limit_mb(path.name) * 1024 ** 2
                    content = file.read(limit + 1)
                if len(content) > limit:
                    raise ValueError(f"文件大小上限为 {file_limit_mb(path.name)} MB。")
                results.append(self.ingest(path.name, content, station))
            except (ValueError, OSError, RuntimeError) as exc:
                results.append({"source": str(relative), "status": "error", "detail": str(exc)})
        return results

    def list_manuals(self) -> list[dict]:
        with closing(self.connect()) as db:
            return [dict(row) for row in db.execute("""
                SELECT id, source, station, pages, text_pages, chunks,
                       (SELECT count(DISTINCT sheet) FROM spreadsheet_locations WHERE doc_id = manuals.id) AS sheets
                FROM manuals WHERE active = 1 ORDER BY station, source
            """)]

    def get_original(self, doc_id: str):
        with closing(self.connect()) as db:
            return db.execute("SELECT source, pdf AS content FROM manuals WHERE id = ?", (doc_id,)).fetchone()

    def encode(self, texts: list[str]) -> list[bytes]:
        try:
            return self.embeddings.documents(texts)
        except Exception as exc:
            logger.warning("文档向量化失败", exc_info=True)
            raise EmbeddingUnavailable("向量模型暂不可用，文件未更新。") from exc

    def backfill_vectors(self, batch_size: int = 32) -> dict:
        if not self.embeddings:
            raise ValueError("请配置 ZHIYUAN_RAG_EMBEDDING_MODEL。")
        model = self.embeddings.model_id
        while True:
            with closing(self.connect()) as db:
                rows = db.execute("""
                    SELECT p.rowid AS id, p.doc_id, p.text
                    FROM passages p LEFT JOIN passage_vectors v
                    ON p.rowid = v.passage_id AND v.model_id = ?
                    WHERE v.passage_id IS NULL ORDER BY p.rowid LIMIT ?
                """, (model, batch_size)).fetchall()
            if not rows:
                return self.vector_status()
            vectors = self.encode([row["text"] for row in rows])
            with closing(self.connect()) as db, db:
                # 原文可能在编码期间被替换。只为仍匹配的片段写入向量。
                db.executemany("""
                    INSERT INTO passage_vectors
                    SELECT p.rowid, ?, ? FROM passages p
                    WHERE p.rowid = ? AND p.doc_id = ? AND p.text = ?
                    ON CONFLICT(passage_id) DO UPDATE SET model_id = excluded.model_id, vector = excluded.vector
                """, [(model, vector, row["id"], row["doc_id"], row["text"])
                      for row, vector in zip(rows, vectors)])

    def vector_status(self) -> dict:
        model = self.embeddings.model_id if self.embeddings else ""
        with closing(self.connect()) as db:
            total, stored = db.execute("""
                SELECT count(*), count(v.passage_id)
                FROM passages p LEFT JOIN passage_vectors v
                ON p.rowid = v.passage_id AND v.model_id = ?
            """, (model,)).fetchone()
        return {"model": self.embeddings.model_name if self.embeddings else None,
                "chunks": total, "indexed": stored}

    @staticmethod
    def fuse(lexical: list[int], semantic: list[int]) -> list[int]:
        scores = {}
        for ranking in (lexical, semantic):
            for rank, chunk in enumerate(ranking, 1):
                scores[chunk] = scores.get(chunk, 0.0) + 1 / (60 + rank)
        return sorted(scores, key=scores.get, reverse=True)

    @staticmethod
    def make_passage(row, retrieved_by: list[str]) -> Passage:
        return Passage(row["text"], {
            "source": row["source"], "page": row["page"], "station": row["station"],
            "doc_id": row["id"], "chunk_id": f'{row["id"]}_{row["chunk"]}',
            "sheet": row["sheet"], "cell_range": row["cell_range"],
            "file_url": f'/api/documents/{row["id"]}/original',
            "pdf_url": f'/api/documents/{row["id"]}/pdf#page={row["page"]}' if row["page"] else None,
            "retrieved_by": retrieved_by,
        })

    def numbered_candidates(self, action: str, station: str) -> list[tuple[int, Passage]]:
        terms = set(search_terms(STATION_CODE.sub("", action)))
        candidates = []
        for passage in self.retrieve(action, top_k=20, station=station):
            step = EXCEL_STEP.match(passage.page_content) if passage.metadata["sheet"] else None
            if step:
                overlap = len(terms.intersection(search_terms(step["action"])))
                if overlap:
                    candidates.append((overlap, passage))
        return sorted(candidates, key=lambda item: item[0], reverse=True)

    def remember_step(self, session_id: str, passage: Passage | None) -> None:
        with closing(self.connect()) as db, db:
            if not passage or not passage.metadata["sheet"] or not EXCEL_STEP.match(passage.page_content):
                db.execute("DELETE FROM session_steps WHERE session_id = ?", (session_id,))
                return
            now = int(time())
            db.execute("DELETE FROM session_steps WHERE expires_at <= ?", (now,))
            db.execute("""
                INSERT INTO session_steps VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    doc_id = excluded.doc_id, passage_id = excluded.passage_id,
                    sheet = excluded.sheet, cell_range = excluded.cell_range, expires_at = excluded.expires_at
            """, (session_id, passage.metadata["doc_id"], int(passage.metadata["chunk_id"].rsplit("_", 1)[1]),
                  passage.metadata["sheet"], passage.metadata["cell_range"], now + settings.session_ttl))

    def last_step(self, session_id: str, station: str) -> Passage | None:
        with closing(self.connect()) as db:
            row = db.execute("""
                SELECT p.rowid AS chunk, p.text, p.page, m.id, m.source, m.station, x.sheet, x.cell_range
                FROM session_steps s JOIN passages p ON p.rowid = s.passage_id AND p.doc_id = s.doc_id
                JOIN manuals m ON m.id = p.doc_id
                JOIN spreadsheet_locations x ON x.passage_id = p.rowid AND x.doc_id = s.doc_id
                    AND x.sheet = s.sheet AND x.cell_range = s.cell_range
                WHERE s.session_id = ? AND s.expires_at > ? AND m.active = 1 AND (? = '' OR m.station = ?)
            """, (session_id, int(time()), station, station)).fetchone()
        return self.make_passage(row, ["session"]) if row and EXCEL_STEP.match(row["text"]) else None

    def adjacent_step(self, action: str, station: str, offset: int, candidates=None) -> tuple[Passage | None, str]:
        if candidates is None:
            candidates = self.numbered_candidates(action, station)
        if not candidates:
            return None, "没有在指导书中定位到你说的当前步骤。请补充工位和操作名称。"
        best = max(item[0] for item in candidates)
        neighbors = {}
        missing = False
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            for overlap, anchor in candidates:
                if overlap != best:
                    continue
                step = EXCEL_STEP.match(anchor.page_content)
                rows = db.execute("""
                    SELECT p.rowid AS chunk, p.text, p.page, m.id, m.source, m.station, x.sheet, x.cell_range
                    FROM passages p JOIN manuals m ON m.id = p.doc_id
                    JOIN spreadsheet_locations x ON x.passage_id = p.rowid
                    WHERE m.active = 1 AND m.id = ? AND x.sheet = ?
                """, (anchor.metadata["doc_id"], anchor.metadata["sheet"]))
                found = False
                for row in rows:
                    neighbor = EXCEL_STEP.match(row["text"])
                    if (neighbor and int(neighbor["number"]) == int(step["number"]) + offset
                            and (int(neighbor["row"]) - int(step["row"])) * offset > 0):
                        found = True
                        passage = self.make_passage(row, ["sequence"])
                        neighbors.setdefault((row["station"], row["text"]), passage)
                missing |= not found
        if not neighbors:
            return None, "定位到了当前步骤，但该工作表没有可确认的相邻编号步骤。"
        if len(neighbors) > 1 or missing:
            return None, "找到多个可能的操作顺序。请补充工位上的具体工序或产品型号。"
        return next(iter(neighbors.values())), ""

    def retrieve(self, query: str, top_k: int = 5, station: str = "") -> list[Passage]:
        if not station:
            codes = station_codes(query)
            if len(codes) == 1:
                station = codes.pop()
                # 工位作为检索条件，避免工位编号本身让封面盖过操作步骤。
                query = STATION_CODE.sub("", query).strip() or station
        common = {"怎么", "如何", "什么", "请问", "一下", "哪些", "是否", "应该", "需要", "可以"}
        terms = list(dict.fromkeys(term for term in search_terms(query) if term not in common))
        alternatives = " OR ".join(f'"{term}"' for term in terms[:120])
        # 文件名和工位可调整排序，正文至少要命中一个检索词。
        match = f"terms : ({alternatives}) AND ({alternatives})"
        limit = max(top_k, settings.rag_candidate_limit) if self.embeddings or self.reranker else top_k
        query_vector = None
        if self.embeddings and query.strip():
            try:
                query_vector = self.embeddings.query(query)
            except Exception:
                logger.warning("问题向量化失败，使用全文检索", exc_info=True)
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            lexical = [row[0] for row in db.execute("""
                SELECT p.rowid
                FROM passages p JOIN manuals m ON m.id = p.doc_id
                WHERE passages MATCH ? AND m.active = 1 AND (? = '' OR m.station = ?)
                ORDER BY bm25(passages, 0, 0, 0, 1.0, 0.25), p.rowid LIMIT ?
            """, (match, station, station, limit))] if terms else []
            semantic = []
            if query_vector is not None:
                vectors = db.execute("""
                    SELECT v.passage_id, v.vector
                    FROM passage_vectors v JOIN passages p ON p.rowid = v.passage_id
                    JOIN manuals m ON m.id = p.doc_id
                    WHERE m.active = 1 AND (? = '' OR m.station = ?)
                    AND v.model_id = ? AND length(v.vector) = ? ORDER BY v.passage_id
                """, (station, station, self.embeddings.model_id, query_vector.nbytes)).fetchall()
                semantic = self.embeddings.rank(query_vector, vectors, settings.rag_vector_min_score, limit)
            ids = self.fuse(lexical, semantic)[:limit if self.reranker else top_k]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            rows = {row["chunk"]: row for row in db.execute(f"""
                SELECT p.rowid AS chunk, p.text, p.page, m.id, m.source, m.station, x.sheet, x.cell_range
                FROM passages p JOIN manuals m ON m.id = p.doc_id
                LEFT JOIN spreadsheet_locations x ON x.passage_id = p.rowid
                WHERE p.rowid IN ({placeholders})
            """, ids)}
        if self.reranker and len(ids) > 1:
            try:
                ranking = self.reranker.rank(query, [rows[chunk]["text"] for chunk in ids])
                ids = [ids[position] for position in ranking]
            except Exception:
                logger.warning("重排模型不可用，使用融合排序", exc_info=True)
        return [self.make_passage(rows[chunk], (["keyword"] if chunk in lexical else [])
                                  + (["semantic"] if chunk in semantic else []))
                for chunk in ids[:top_k]]
