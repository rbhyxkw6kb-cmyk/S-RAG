"""Exercise the single-process API with synthetic manuals and no external network."""

import asyncio
import json
import math
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import httpx
import pymupdf
from pydantic import SecretStr

from app.core.config import settings
from app.main import app
from app.services.instruction_chat import InstructionChat

STATIONS = [f"WS{number}" for number in range(1100, 1140)]
QUERY = "确认标牌完整"


def sample_pdf(text: str) -> bytes:
    with pymupdf.open() as document:
        document.new_page().insert_text((60, 60), text, fontname="china-s")
        return document.tobytes()


class SimulatedModel:
    def __init__(self):
        self.mode = "healthy"
        self.calls = 0

    async def respond(self, request):
        self.calls += 1
        if self.mode == "rate_limited":
            return httpx.Response(429, headers={"Retry-After": "1"})
        await asyncio.sleep(5 if self.mode == "timeout" else 0.045)
        content = json.loads(request.content)["messages"][0]["content"]
        if "检索规划器" in content:
            result = {"intent": "lookup", "search_query": QUERY}
        else:
            result = {"paragraphs": [{"text": "请确认标牌完整。", "quote": QUERY, "source": 1}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})


async def load(client, model, label, count, *, interval=0):
    calls_before = model.calls
    start = perf_counter()

    async def request(index):
        if interval:
            await asyncio.sleep(index * interval)
        started = perf_counter()
        result = await client.post("/api/chat", json={
            "query": QUERY, "station": STATIONS[index % len(STATIONS)], "stream": False,
        })
        if result.status_code == 429:
            kind = "http_429"
        else:
            body = result.raise_for_status().json()
            if not body["citations"]:
                raise AssertionError(f"Missing synthetic evidence in {label}")
            kind = body.get("degraded_reason", body["mode"])
        return kind, (perf_counter() - started) * 1000

    results = await asyncio.gather(*(request(index) for index in range(count)))
    outcomes = Counter(name for name, _ in results)
    served = sorted(duration for name, duration in results if name != "http_429")
    p95 = round(served[math.ceil(len(served) * 0.95) - 1], 1) if served else None
    print(json.dumps({
        "scenario": label, "requests": count, "elapsed_s": round(perf_counter() - start, 2),
        "served_p95_ms": p95, "outcomes": outcomes, "model_requests": model.calls - calls_before,
    }, ensure_ascii=False))
    return outcomes


async def scenarios(root: Path):
    model = SimulatedModel()
    with ExitStack() as stack:
        config = {
            "knowledge_db": str(root / "synthetic.sqlite3"),
            "dataset_dir": str(root / "dataset"),
            "staff_password": None,
            "admin_password": None,
            "llm_base_url": "https://api.deepseek.com",
            "llm_model_name": "deepseek-flash",
            "deepseek_api_key": SecretStr("synthetic-benchmark"),
            "llm_budget_seconds": 0.6,
            "llm_breaker_cooldown": 1.5,
            "llm_requests_per_minute": 0,
        }
        for name, value in config.items():
            stack.enter_context(patch.object(settings, name, value))
        stack.enter_context(patch.object(InstructionChat, "create_client",
                                         return_value=httpx.AsyncClient(transport=httpx.MockTransport(model.respond))))
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://benchmark",
        ) as client:
            data = sample_pdf(f"{QUERY}。检查后记录完成情况。")
            revised = sample_pdf(f"{QUERY}。检查结束后填写记录。")
            for station in STATIONS:
                app.state.library.ingest("manual.pdf", data, station)
                folder = root / "dataset" / station
                folder.mkdir(parents=True)
                (folder / "manual.pdf").write_bytes(revised)
            print(json.dumps({"stations": len(STATIONS), "model_budget_s": settings.llm_budget_seconds,
                              "cooldown_s": settings.llm_breaker_cooldown,
                              "vectors": app.state.library.vector_status()}))

            await load(client, model, "shift_change", 200, interval=0.05)
            scan, _ = await asyncio.gather(client.post("/api/documents/scan"),
                                           load(client, model, "queries_during_import", 40, interval=0.02))
            assert len(scan.raise_for_status().json()["results"]) == len(STATIONS)
            assert all(item["status"] == "ready" for item in scan.json()["results"])
            await load(client, model, "instant_burst", 200)

            model.mode = "timeout"
            await load(client, model, "upstream_timeout", 40, interval=0.04)
            status = (await client.get("/health")).raise_for_status().json()["load"]["model"]
            assert status["state"] == "open", status
            model.mode = "healthy"
            await asyncio.sleep(settings.llm_breaker_cooldown + 0.05)
            assert (await load(client, model, "recovery_probe", 1))["generated"] == 1

            model.mode = "rate_limited"
            await load(client, model, "upstream_429", 40, interval=0.02)


def main():
    with tempfile.TemporaryDirectory() as folder:
        asyncio.run(scenarios(Path(folder)))


if __name__ == "__main__":
    main()
