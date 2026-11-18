import asyncio
import json
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import httpx
from pydantic import SecretStr

from app.core.config import settings
from app.main import app
from app.services.instruction_chat import InstructionChat
from test_instruction_workflow import make_pdf


class ResilienceTests(unittest.IsolatedAsyncioTestCase):
    def configure(self, folder, handler, **overrides):
        configuration = {
            "knowledge_db": str(Path(folder) / "test.sqlite3"),
            "rag_embedding_model": "",
            "rag_reranker_model": "",
            "staff_password": None,
            "admin_password": None,
            "llm_base_url": "https://api.deepseek.com",
            "llm_model_name": "deepseek-flash",
            "deepseek_api_key": SecretStr("synthetic-test"),
            "llm_budget_seconds": 2,
            "llm_max_inflight": 2,
            "llm_requests_per_minute": 0,
            "llm_queue_seconds": 0,
            "llm_breaker_failures": 2,
            "llm_breaker_cooldown": 10,
            "query_max_inflight": 4,
            "query_max_per_station": 1,
        }
        for key, value in (configuration | overrides).items():
            setting = patch.object(settings, key, value)
            setting.start()
            self.addCleanup(setting.stop)
        factory = patch.object(InstructionChat, "create_client", return_value=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ))
        factory.start()
        self.addCleanup(factory.stop)

    @staticmethod
    def upstream_success(request):
        payload = json.loads(request.content)
        if "检索规划器" in payload["messages"][0]["content"]:
            response = {"intent": "lookup", "search_query": "确认标牌完整"}
        else:
            response = {"paragraphs": [{"text": "应确认标牌完整。", "source": 1, "quote": "确认标牌完整"}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)}}]})

    @staticmethod
    async def ask(client, station="WS1001"):
        return await client.post("/api/chat", json={"query": "确认标牌完整", "station": station, "stream": False})

    @staticmethod
    def seed():
        document = make_pdf("确认标牌完整")
        for station in ("WS1001", "WS1002", "WS1003"):
            app.state.library.ingest("test.pdf", document, station)

    async def test_failure_opens_breaker_and_admits_one_recovery_probe(self):
        calls = 0
        recovering = False
        probe_started = asyncio.Event()
        finish_probe = asyncio.Event()

        async def handler(request):
            nonlocal calls
            calls += 1
            if not recovering:
                return httpx.Response(503)
            if not probe_started.is_set():
                probe_started.set()
                await finish_probe.wait()
            return self.upstream_success(request)

        with tempfile.TemporaryDirectory() as folder:
            self.configure(folder, handler)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.seed()
                for _ in range(2):
                    fallback = (await self.ask(client)).json()
                    self.assertEqual(fallback["mode"], "sources")
                    self.assertEqual(fallback["citations"][0]["page"], 2)
                skipped = (await self.ask(client)).json()
                self.assertEqual(skipped["degraded_reason"], "model_circuit")
                self.assertEqual(calls, 2)
                status = (await client.get("/health")).json()["load"]
                self.assertEqual(status["model"]["state"], "open")
                self.assertEqual(status["outcomes"]["degraded"], 3)

                recovering = True
                app.state.chat.model_guard.open_until = monotonic() - 1
                probe = asyncio.create_task(self.ask(client))
                try:
                    await asyncio.wait_for(probe_started.wait(), timeout=2)
                    concurrent = (await self.ask(client, "WS1002")).json()
                    self.assertEqual(concurrent["degraded_reason"], "model_circuit")
                    self.assertEqual(calls, 3)
                finally:
                    finish_probe.set()
                    response = await probe
                self.assertEqual(response.json()["mode"], "generated")
                self.assertEqual(calls, 4)
                self.assertEqual((await client.get("/health")).json()["load"]["model"]["state"], "closed")

    async def test_station_limit_and_upstream_capacity_preserve_other_queries(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(request):
            entered.set()
            await release.wait()
            return self.upstream_success(request)

        with tempfile.TemporaryDirectory() as folder:
            self.configure(folder, handler, llm_max_inflight=1)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.seed()
                pending = asyncio.create_task(self.ask(client))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    limited = await self.ask(client)
                    self.assertEqual(limited.status_code, 429)
                    self.assertEqual(limited.headers["Retry-After"], "1")
                    remaining = (await self.ask(client, "WS1002")).json()
                    self.assertEqual(remaining["degraded_reason"], "model_capacity")
                    self.assertEqual(remaining["citations"][0]["station"], "WS1002")
                finally:
                    release.set()
                    await pending
                self.assertEqual(app.state.queries.active, 0)
                self.assertEqual(app.state.chat.model_guard.status()["in_flight"], 0)

    async def test_shared_model_budget_includes_previous_model_stage(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.07)
            else:
                await asyncio.sleep(0.20)
            return self.upstream_success(request)

        with tempfile.TemporaryDirectory() as folder:
            self.configure(folder, handler, llm_budget_seconds=0.18)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.seed()
                response = (await self.ask(client)).json()
                self.assertEqual(response["degraded_reason"], "model_budget")
                self.assertEqual(response["citations"][0]["page"], 2)
                self.assertEqual(calls, 2)
                self.assertEqual(app.state.chat.model_guard.status()["in_flight"], 0)

    async def test_provider_rate_limit_opens_breaker_immediately(self):
        async def handler(_):
            return httpx.Response(429, headers={"Retry-After": "90"})

        with tempfile.TemporaryDirectory() as folder:
            self.configure(folder, handler)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.seed()
                result = (await self.ask(client)).json()
                self.assertEqual(result["mode"], "sources")
                guard = app.state.chat.model_guard
                self.assertGreater(guard.open_until - monotonic(), 85)
                self.assertEqual((await self.ask(client)).json()["degraded_reason"], "model_circuit")
                self.assertEqual(guard.calls, 1)

    async def test_configured_upstream_rate_limit_degrades_without_calling_model_again(self):
        with tempfile.TemporaryDirectory() as folder:
            self.configure(folder, self.upstream_success, llm_requests_per_minute=1)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.seed()
                result = (await self.ask(client)).json()
                self.assertEqual(result["degraded_reason"], "model_rate")
                self.assertTrue(result["citations"])
                self.assertEqual(app.state.chat.model_guard.calls, 1)

    async def test_slow_import_has_independent_admission(self):
        active = threading.Event()
        release = threading.Event()

        def blocked_scan():
            with closing(app.state.library.connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                active.set()
                release.wait(timeout=4)
            return []

        with tempfile.TemporaryDirectory() as folder:
            self.configure(folder, self.upstream_success)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                self.seed()
                with patch.object(app.state.library, "scan", blocked_scan):
                    task = asyncio.create_task(client.post("/api/documents/scan"))
                    try:
                        self.assertTrue(await asyncio.to_thread(active.wait, 2))
                        other_import = await client.post("/api/documents/scan")
                        self.assertEqual(other_import.status_code, 429)
                        query = await asyncio.wait_for(self.ask(client), timeout=2)
                        self.assertEqual(query.json()["mode"], "generated")
                    finally:
                        release.set()
                        self.assertEqual((await task).status_code, 200)
