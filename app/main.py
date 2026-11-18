import json
import logging
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import anyio
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.core.access import access_role, require_admin
from app.core.config import settings
from app.core.resilience import QueryAdmission, QueryBusy
from app.services.instruction_chat import InstructionChat
from app.services.instruction_library import EmbeddingUnavailable, InstructionLibrary, file_limit_mb, station_codes


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.library = InstructionLibrary(settings.knowledge_db)
    app.state.queries = QueryAdmission(settings.query_max_inflight, settings.query_max_per_station)
    app.state.ingest = QueryAdmission(1, 1)
    app.state.ingest_worker = anyio.CapacityLimiter(1)
    app.state.outcomes = Counter()
    async with InstructionChat.create_client() as client:
        app.state.chat = InstructionChat(app.state.library, client)
        yield


app = FastAPI(title=settings.app_name, version="1.1.0", lifespan=lifespan)
web_dir = Path(__file__).resolve().parent / "web"
app.mount("/assets", StaticFiles(directory=web_dir), name="assets")
logger = logging.getLogger("uvicorn.error")


@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = uuid.uuid4().hex
    started = perf_counter()
    origin = request.headers.get("Origin")
    if request.method == "POST" and origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        response = JSONResponse({"detail": "此接口仅接受同源浏览器请求。"}, status_code=403)
    else:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    if request.url.path.startswith("/api/"):
        route = request.scope.get("route")
        logger.info("request_id=%s method=%s route=%s status=%s duration_ms=%d",
                    request_id, request.method, getattr(route, "path", "unknown"), response.status_code,
                    (perf_counter() - started) * 1000)
        response.headers["Cache-Control"] = "no-store"
    return response


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=100)
    station: str = Field(default="", max_length=80)
    stream: bool = True

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value):
        return value.strip() if isinstance(value, str) else value


@app.get("/", dependencies=[Depends(access_role)])
def index():
    return FileResponse(web_dir / "index.html")


@app.get("/health")
async def health(request: Request):
    state = request.app.state
    return {"status": "ok", "service": settings.app_name, "model_configured": settings.llm_configured,
            "model_provider": "deepseek" if settings.uses_deepseek else None,
            "vectors": await run_in_threadpool(state.library.vector_status),
            "load": {"queries": state.queries.status(), "ingest": state.ingest.status(),
                     "model": state.chat.model_guard.status(), "outcomes": dict(state.outcomes)}}


def query_scope(query: str, station: str) -> str:
    codes = station_codes(query)
    return (station or (codes.pop() if len(codes) == 1 else "")).casefold()


def busy(message: str) -> HTTPException:
    return HTTPException(429, message, headers={"Retry-After": "1"})


@app.get("/api/access")
def access(role: str = Depends(access_role)):
    return {"can_import": role == "admin"}


@app.get("/api/documents", dependencies=[Depends(access_role)])
def documents(request: Request):
    return request.app.state.library.list_manuals()


@app.post("/api/documents/upload", dependencies=[Depends(require_admin)])
async def upload(request: Request, file: UploadFile, station: str = Form(default="", max_length=80)):
    limit_mb = file_limit_mb(file.filename or "")
    limit = limit_mb * 1024 ** 2
    state = request.app.state
    try:
        with state.ingest.reserve("import"):
            content = await file.read(limit + 1)
            if len(content) > limit:
                raise HTTPException(413, f"文件大小上限为 {limit_mb} MB。")
            return await anyio.to_thread.run_sync(
                state.library.ingest, file.filename or "", content, station, limiter=state.ingest_worker,
            )
    except QueryBusy as exc:
        raise busy("已有指导书正在导入，请稍后再试。") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except EmbeddingUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/documents/scan", dependencies=[Depends(require_admin)])
async def scan(request: Request):
    state = request.app.state
    try:
        with state.ingest.reserve("import"):
            return {"results": await anyio.to_thread.run_sync(state.library.scan, limiter=state.ingest_worker)}
    except QueryBusy as exc:
        raise busy("已有指导书正在导入，请稍后再试。") from exc


@app.get("/api/documents/{doc_id}/pdf", dependencies=[Depends(access_role)])
def pdf(request: Request, doc_id: str):
    document = request.app.state.library.get_original(doc_id)
    if document is None or Path(document["source"]).suffix.lower() != ".pdf":
        raise HTTPException(404, "未找到 PDF。")
    return Response(document["content"], media_type="application/pdf", headers={
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(document['source'], safe='')}",
        "X-Content-Type-Options": "nosniff",
    })


@app.get("/api/documents/{doc_id}/original", dependencies=[Depends(access_role)])
def original(request: Request, doc_id: str):
    document = request.app.state.library.get_original(doc_id)
    if document is None:
        raise HTTPException(404, "未找到原文件。")
    return Response(document["content"], media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(document['source'], safe='')}",
        "X-Content-Type-Options": "nosniff",
    })


@app.get("/api/documents/{doc_id}/figures/{figure_id}", dependencies=[Depends(access_role)])
def figure(request: Request, doc_id: str, figure_id: int):
    picture = request.app.state.library.get_figure(doc_id, figure_id)
    if picture is None:
        raise HTTPException(404, "未找到图例。")
    return Response(picture["png"], media_type="image/png", headers={"X-Content-Type-Options": "nosniff"})


@app.post("/api/search", dependencies=[Depends(access_role)])
async def search(request: Request, query: str = Query(min_length=1, max_length=1000),
           top_k: int = Query(default=5, ge=1, le=30), station: str = Query(default="", max_length=80)):
    state = request.app.state
    try:
        with state.queries.reserve(query_scope(query, station.strip())):
            results = await state.chat.run_local(state.library.retrieve, query, top_k, station.strip())
    except QueryBusy as exc:
        raise busy("当前查询较多，请稍后再试。") from exc
    return {"query": query, "results": [
        {"content": result.page_content, "metadata": result.metadata} for result in results
    ]}


@app.post("/api/chat", dependencies=[Depends(access_role)])
async def chat(request: Request, req: ChatRequest):
    state = request.app.state
    session_id = req.session_id or str(uuid.uuid4())
    station = req.station.strip()
    try:
        with state.queries.reserve(query_scope(req.query, station)):
            result = await state.chat.answer(req.query, station, session_id)
            if result["citations"]:
                await state.chat.run_local(state.library.add_citation_figures, result["citations"])
    except QueryBusy as exc:
        raise busy("当前查询较多，请稍后再试。") from exc
    result["session_id"] = session_id
    state.outcomes[result["mode"]] += 1
    if "degraded_reason" in result:
        state.outcomes["degraded"] += 1

    if not req.stream:
        return result

    async def events():
        for event in ({"type": "token", "content": result["answer"]}, {"type": "done", **result}):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False)
