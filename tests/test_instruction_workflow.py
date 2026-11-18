import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pymupdf
from openpyxl import Workbook
from pydantic import SecretStr

from app.core.config import settings
from app.main import app
from app.services.instruction_chat import InstructionChat
from app.services.instruction_library import InstructionLibrary


def make_pdf(instruction: str) -> bytes:
    with pymupdf.open() as document:
        document.new_page().insert_text((60, 60), "Station cover")
        document.new_page().insert_text((60, 60), instruction, fontname="china-s")
        document.new_page()
        return document.tobytes()


class InstructionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for name, value in (("rag_embedding_model", ""), ("rag_reranker_model", ""), ("llm_evidence_only", False),
                            ("llm_base_url", "http://localhost:8001/v1"),
                            ("staff_password", None), ("admin_password", None)):
            setting = patch.object(settings, name, value)
            setting.start()
            self.addCleanup(setting.stop)

    async def test_staff_can_query_but_only_admin_can_import(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(settings, "knowledge_db", str(Path(folder) / "manuals.sqlite3")), \
             patch.object(settings, "dataset_dir", folder), \
             patch.object(settings, "staff_password", SecretStr("staff-test")), \
             patch.object(settings, "admin_password", SecretStr("admin-test")):
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.assertEqual((await client.get("/api/documents")).status_code, 401)
                staff = ("staff", "staff-test")
                admin = ("admin", "admin-test")
                self.assertEqual((await client.get("/api/documents", auth=staff)).status_code, 200)
                self.assertEqual((await client.post("/api/documents/scan", auth=staff)).status_code, 403)
                self.assertEqual((await client.post("/api/documents/scan", auth=admin)).status_code, 200)
                foreign = await client.post("/api/documents/scan", auth=admin, headers={"Origin": "https://example.com"})
                self.assertEqual(foreign.status_code, 403)

    async def test_mixed_folder_and_excel_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            station = root / "装配工位"
            station.mkdir()
            workbook = Workbook()
            workbook.active.title = "作业步骤"
            workbook.active["A1"] = "作业分解指导书"
            workbook.active.merge_cells("B2:D2")
            workbook.active["B2"] = "启动前校准扭矩仪"
            workbook.active["B4"] = "确认接插件方向"
            workbook.active["D4"] = "读取绿色标记"
            workbook.active["A6"] = "1. 贴上安全标签"
            workbook.active["F7"] = "标签边缘应贴平"
            workbook.active["A8"] = "2. 如果标识缺失，记录批次号"
            workbook.active["A9"] = "3. 将记录交给班组长"
            workbook.active["XFD38"].number_format = "0.00"
            workbook.create_sheet("检查记录")["A1"] = "扫码记录批次号"
            template = workbook.create_sheet("旧模板")
            template.sheet_state = "hidden"
            template["A1"] = "确认接插件方向"
            stream = io.BytesIO()
            workbook.save(stream)
            workbook.close()
            original = stream.getvalue()
            (root / "SAMPLE_WS1001_指导书.XLSX").write_bytes(original)
            (station / "补充说明.pdf").write_bytes(make_pdf("确认接插件方向，参见箭头。"))
            (station / "~$指导书.xlsx").write_bytes(b"excel lockfile")
            with patch.object(settings, "knowledge_db", str(root / "manuals.sqlite3")), \
                 patch.object(settings, "dataset_dir", str(root)), \
                 patch.object(settings, "llm_model_name", ""):
                async with app.router.lifespan_context(app), httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client:
                    scan = (await client.post("/api/documents/scan")).json()
                    self.assertEqual(len(scan["results"]), 2)
                    self.assertTrue(all(item["status"] == "ready" for item in scan["results"]), scan)
                    manuals = (await client.get("/api/documents")).json()
                    excel = next(item for item in manuals if item["sheets"])
                    self.assertEqual(excel["sheets"], 2)
                    self.assertEqual(excel["station"], "WS1001")
                    result = await client.post("/api/chat", json={
                        "query": "接插件方向", "stream": False,
                    })
                    citations = result.json()["citations"]
                    cite = next(item for item in citations if item["sheet"])
                    self.assertEqual((cite["sheet"], cite["cell_range"]), ("作业步骤", "B4:D4"))
                    self.assertTrue(any(item["page"] == 2 for item in citations))
                    scoped = await client.post("/api/chat", json={
                        "query": "WS1001的接插件方向", "stream": False,
                    })
                    scoped_citations = scoped.json()["citations"]
                    self.assertEqual([item["station"] for item in scoped_citations], ["WS1001"])
                    step = await client.post("/api/chat", json={
                        "query": "贴上安全标签", "station": "WS1001", "stream": False,
                    })
                    step_cite = step.json()["citations"][0]
                    self.assertEqual(step_cite["cell_range"], "A6:F7")
                    self.assertIn("标签边缘应贴平", step_cite["snippet"])
                    async def sequence(question, session_id=None, scope="WS1001"):
                        return (await client.post("/api/chat", json={
                            "query": question, "station": scope, "session_id": session_id, "stream": False,
                        })).json()

                    for query in ("贴上安全标签后应该干什么", "安全标签贴好下一步是什么",
                                  "贴上安全标签，接下来做什么？", "贴上安全标签后下一步是什么？"):
                        with self.subTest(query=query):
                            following = await sequence(query)
                            self.assertEqual(following["mode"], "sequence")
                            self.assertIn("如果标识缺失，记录批次号", following["answer"])
                            self.assertEqual(following["citations"][0]["cell_range"], "A8")
                    previous = await sequence("记录批次号之前做什么？")
                    self.assertEqual(previous["mode"], "sequence")
                    self.assertEqual(previous["citations"][0]["cell_range"], "A6:F7")
                    self.assertEqual((await sequence("将记录交给班组长后做什么"))["mode"], "no_match")
                    self.assertEqual((await sequence("下一步做什么"))["mode"], "no_match")
                    self.assertEqual((await sequence("下一步是什么"))["mode"], "no_match")
                    responses = [
                        '{"intent":"next","search_query":"贴上安全标签"}',
                        '{"paragraphs":[{"text":"贴好安全标签后，只有发现标识缺失时才需要记录批次号。",'
                        '"source":1,"quote":"如果标识缺失，记录批次号"}]}',
                        '{"paragraphs":[{"text":"下一步，将记录交给班组长。",'
                        '"source":1,"quote":"将记录交给班组长"}]}',
                    ]
                    external_post = httpx.AsyncClient.post

                    async def fake_deepseek(http_client, url, *args, **kwargs):
                        if url != "https://api.deepseek.com/chat/completions":
                            return await external_post(http_client, url, *args, **kwargs)
                        payload = kwargs["json"]
                        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-token")
                        self.assertEqual(payload["model"], "deepseek-flash")
                        self.assertEqual(payload["response_format"], {"type": "json_object"})
                        if len(responses) == 2:
                            evidence = payload["messages"][1]["content"]
                            self.assertIn("如果标识缺失，记录批次号", evidence)
                            self.assertNotIn("标签边缘应贴平", evidence)
                        if len(responses) == 1:
                            evidence = payload["messages"][1]["content"]
                            self.assertIn("将记录交给班组长", evidence)
                            self.assertNotIn("如果标识缺失", evidence)
                        return httpx.Response(200, request=httpx.Request("POST", url), json={
                            "choices": [{"message": {"content": responses.pop(0)}}],
                        })

                    with patch.object(settings, "llm_base_url", "https://api.deepseek.com"), \
                         patch.object(settings, "llm_model_name", "deepseek-flash"), \
                         patch.object(settings, "deepseek_api_key", SecretStr("test-token")), \
                         patch.object(httpx.AsyncClient, "post", fake_deepseek):
                        composed = await sequence("安全标签贴完之后还有事吗？")
                        self.assertEqual(composed["mode"], "generated")
                        self.assertIn("只有发现标识缺失时", composed["answer"])
                        self.assertEqual(composed["citations"][0]["cell_range"], "A8")
                        # 模拟新服务进程，已经回答的步骤仍可作为下一轮的定位依据。
                        with patch.object(app.state, "chat", InstructionChat(InstructionLibrary(settings.knowledge_db))):
                            session = composed["session_id"]
                            self.assertEqual((await sequence("再下一步呢", session, "WS1000"))["mode"], "no_match")
                            self.assertEqual((await sequence("再下一步呢"))["mode"], "no_match")
                            continued = await sequence("再下一步呢", session)
                            self.assertEqual(continued["mode"], "generated")
                            self.assertIn("将记录交给班组长", continued["answer"])
                            self.assertEqual(continued["citations"][0]["cell_range"], "A9")
                        self.assertEqual(responses, [])
                    download = await client.get(cite["file_url"])
                    self.assertEqual(download.content, original)
                    self.assertTrue(download.headers["Content-Disposition"].startswith("attachment"))
                    invalid = await client.post("/api/documents/upload", files={"file": ("invalid.xlsx", b"bad")})
                    self.assertEqual(invalid.status_code, 422)

    async def test_import_scope_citation_and_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch.object(settings, "knowledge_db", str(path / "manuals.sqlite3")), \
                 patch.object(settings, "dataset_dir", str(path)), \
                 patch.object(settings, "llm_model_name", ""):
                async with app.router.lifespan_context(app), httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client:
                    await self.check_workflow(client, path)

    async def check_workflow(self, client, dataset):
        async def upload(content, station):
            return await client.post(
                "/api/documents/upload", data={"station": station},
                files={"file": ("装配作业.pdf", content, "application/pdf")},
            )

        async def ask(query="装配前怎么检查扭矩？", station="A01", stream=False):
            return await client.post("/api/chat", json={"query": query, "station": station, "stream": stream})

        self.assertEqual((await client.get("/")).status_code, 200)
        original = make_pdf("装配前检查扭矩，设定 10 Nm。")
        result = await upload(original, "A01")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["text_pages"], 2)
        self.assertEqual((await upload(original, "A01")).json()["status"], "unchanged")
        self.assertEqual((await upload(make_pdf("装配前检查扭矩，设定 20 Nm。"), "B02")).status_code, 200)
        self.assertEqual((await upload(b"invalid", "A01")).status_code, 422)

        response = (await ask()).json()
        self.assertEqual(response["mode"], "sources")
        self.assertEqual(len(response["citations"]), 1)
        citation = response["citations"][0]
        self.assertEqual((citation["page"], citation["station"]), (2, "A01"))
        self.assertIn("10", citation["snippet"])
        self.assertTrue(citation["pdf_url"].endswith("#page=2"))
        self.assertEqual((await ask(station="C03")).json()["mode"], "no_match")
        self.assertEqual((await ask(query="等离子清洗")).json()["mode"], "no_match")
        self.assertIn("data: [DONE]", (await ask(stream=True)).text)

        self.assertEqual((await upload(make_pdf("装配前检查扭矩，设定 15 Nm。"), "A01")).status_code, 200)
        self.assertIn("15", (await ask()).json()["citations"][0]["snippet"])
        self.assertEqual((await client.get(citation["pdf_url"].split("#")[0])).content, original)

        station_dir = dataset / "C03"
        station_dir.mkdir()
        (station_dir / "guide.PDF").write_bytes(make_pdf("使用扫码枪记录工件编号。"))
        self.assertEqual((await client.post("/api/documents/scan")).json()["results"][0]["status"], "ready")
        self.assertEqual((await ask(query="扫码枪", station="C03")).json()["citations"][0]["page"], 2)
        self.assertEqual(len((await client.get("/api/documents")).json()), 3)

        with patch.object(settings, "llm_model_name", "local-test"):
            with patch.object(app.state.chat, "generate", AsyncMock(return_value="检查扭矩 [1]。")):
                self.assertEqual((await ask()).json()["mode"], "generated")
            with patch.object(app.state.chat, "generate", AsyncMock(return_value="无法回答")):
                self.assertEqual((await ask()).json()["mode"], "sources")
            with patch.object(settings, "llm_evidence_only", True):
                with patch.object(app.state.chat, "generate", AsyncMock(return_value='{"sources":[1]}')):
                    grounded = (await ask()).json()
                    self.assertIn("设定 15 Nm", grounded["answer"])
                    self.assertEqual(grounded["citations"][0]["page"], 2)
                with patch.object(app.state.chat, "generate", AsyncMock(return_value='{"sources":[99]}')):
                    self.assertEqual((await ask()).json()["mode"], "sources")
