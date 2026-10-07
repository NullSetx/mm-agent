"""FastAPI 网关（文档 §4.3：前端和 Agent 只认 /api/*）。

职责：
- 工具发现：启动和 POST /api/tools/refresh 时并发拉各节点 GET /tools，
  合成「工具名 → 节点」路由表。加新工具不需要改这里（文档 §5 可扩展点）。
- 转发：POST /api/invoke 按路由表透传到所属节点，错误按约定映射
  （未知工具 404 / 下游不可达 502 / 超时 504）。
- 对话：POST /api/chat 编排 LangChain Agent（llm_node.agent），可带图，
  支持按 session_id 续聊（内存会话）。

启动：uvicorn llm_node.gateway:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field

from common.config import DATA_DIR, TOOL_TIMEOUT, mock_enabled
from common.schemas import InvokeRequest, InvokeResponse, ToolSpec, ToolList
from llm_node import agent, llm
from llm_node.sessions import SessionStore

#: 测试台静态页（浏览器打开 http://<网关>:8000/ 即是）
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: 接进网关的工具节点（vision 两节点 + kb 知识库）。
#: 新增节点：config.HOSTS/PORTS 加键 + 这里加名字
VISION_NODES: tuple[str, ...] = ("vision-fast", "vision-heavy", "kb")

_DISCOVER_TIMEOUT = 5.0  # 发现 /health 探测的兜底超时（秒）
_HEALTH_TIMEOUT = 3.0
_CONNECT_TIMEOUT = 5.0

#: 会话历史保留的最大消息数（含工具消息）
MAX_HISTORY_MESSAGES = 108

#: 需要 CORS。默认放开（内网演示）；公网部署时应改成具体来源，逗号分隔
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]

#: SSE 响应头：关掉中间层缓冲，否则流会被攒成一坨再吐出来
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: 追问轮（本轮没传新图、会话里有图）注入给模型的提示：不然小模型会误判
#: "本轮没图 = 没图可用"，让用户重传一张明明已经在会话里的图
_FOLLOWUP_IMAGE_HINT = (
    "（会话中已有一张此前上传的图片；调用 ocr/stylize 等图片工具时会自动使用它，"
    "无需用户重新上传）"
)


def node_url(node: str) -> str:
    """拼视觉节点基址，如 http://192.168.1.101:8101。

    不用 common.config.base_url：它对 HOSTS 有而 PORTS 没有的键会 KeyError、
    未知键会静默错路由到 vision-heavy（common/config.py 现成问题，已报群里）。
    """
    from common.config import HOSTS, PORTS  # 调用时取值，测试改环境变量才生效

    return f"http://{HOSTS[node]}:{PORTS[node]}"


#: 带图请求网关自动预取的分析类工具（并行调用）。
#: 读字（ocr）、生成（stylize）不预取：ocr 慢且多数图没文字，由 LLM 按系统
#: 提示的决策策略自主调用（agent._system_prompt 规则 4）。
ANALYSIS_TOOLS: tuple[str, ...] = ("detect", "classify")


async def _prefetch_observations(
    http: httpx.AsyncClient,
    catalog: ToolCatalog,
    image: str,
) -> tuple[str, list[dict[str, Any]]]:
    """带图请求先确定性调用分析类工具，把润色后的观察文本注入上下文。

    成员确认的架构：图片分析不赌 LLM 自觉调工具——网关直接并行调用
    当前可用的分析类工具（render_for_llm 润色），模型拿到的就是
    整理好的观察，照着回答即可。
    读字（ocr）、生成（stylize）不预取，由模型按需自主调用。
    返回 (观察文本, 预取的工具调用记录)。
    """
    available = {s.name for s in catalog.specs()}
    targets = [n for n in ANALYSIS_TOOLS if n in available]
    if not targets:
        return "（当前未接入任何图片分析工具，无法分析图片内容。）", []

    async def run(name: str):
        resp = await invoke_tool(http, catalog, name, image, {})
        return name, resp

    results = await asyncio.gather(*(run(n) for n in targets))
    records: list[dict[str, Any]] = []
    blocks: list[str] = []
    for name, resp in results:
        records.append(
            {"tool": name, "ok": resp.ok, "result": resp.result, "error": resp.error}
        )
        blocks.append(f"[{name} 观察] {agent.render_for_llm(name, resp.result, resp.error)}")
    return "\n".join(blocks), records


class GatewayError(RuntimeError):
    """网关层错误。status_code 按文档 §4.3：404 未知工具 / 502 下游不可用 / 504 超时。"""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class DuplicateToolError(RuntimeError):
    """跨节点工具重名（文档 §5：网关发现重名直接报错）。"""


# ---------------------------------------------------------------- 工具目录

class ToolCatalog:
    """工具名 → (节点名, ToolSpec) 路由表。

    refresh 全量重建：只保留本轮应答节点的工具（失联节点的工具随之摘除，
    在 refresh 报告里标注）；发现跨节点重名时整体失败、保留上一份目录。
    """

    def __init__(self) -> None:
        self._routes: dict[str, tuple[str, ToolSpec]] = {}

    def get(self, tool: str) -> tuple[str, ToolSpec] | None:
        return self._routes.get(tool)

    def specs(self) -> list[ToolSpec]:
        return [spec for _, spec in sorted(self._routes.values(), key=lambda p: p[1].name)]

    def names(self) -> list[str]:
        return sorted(self._routes)

    def swap(self, routes: dict[str, tuple[str, ToolSpec]]) -> None:
        self._routes = routes


async def refresh_catalog(http: httpx.AsyncClient, catalog: ToolCatalog) -> dict[str, Any]:
    """并发发现所有节点的工具清单，重建路由表。

    Returns:
        每个节点的发现报告：{node: {ok, tools, error, elapsed_ms}}。

    Raises:
        DuplicateToolError: 跨节点工具重名。此时目录保持原样（不 swap）。
    """
    report: dict[str, Any] = {}
    scanned: dict[str, tuple[str, ToolSpec]] = {}

    for node in VISION_NODES:
        started = time.perf_counter()
        try:
            resp = await http.get(f"{node_url(node)}/tools", timeout=_DISCOVER_TIMEOUT)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            resp.raise_for_status()
            body = ToolList.model_validate(resp.json())
            for spec in body.tools:
                if spec.name in scanned:
                    raise DuplicateToolError(
                        f"工具 {spec.name!r} 同时出现在 "
                        f"{scanned[spec.name][0]} 和 {node}，拒绝刷新（工具名必须全局唯一）"
                    )
                scanned[spec.name] = (node, spec)
            report[node] = {
                "ok": True,
                "tools": [s.name for s in body.tools],
                "error": None,
                "elapsed_ms": elapsed,
            }
        except DuplicateToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - 单节点失联只降级，不阻塞其他节点
            report[node] = {
                "ok": False,
                "tools": [],
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }

    catalog.swap(scanned)
    return report


# ---------------------------------------------------------------- 转发

async def invoke_tool(
    http: httpx.AsyncClient,
    catalog: ToolCatalog,
    tool: str,
    image: str | None = None,
    params: dict[str, Any] | None = None,
) -> InvokeResponse:
    """按路由表调用一个工具。404/502/504 抛 GatewayError（不走 HTTP 异常），
    供 /api/invoke 与 Agent 工具两条路径共用。"""
    route = catalog.get(tool)
    if route is None:
        raise GatewayError(404, f"未知工具 {tool!r}，可用工具：{catalog.names()}")
    node, _spec = route

    req = InvokeRequest(tool=tool, image=image, params=params or {})
    try:
        resp = await http.post(
            f"{node_url(node)}/invoke",
            json=req.model_dump(),
            timeout=httpx.Timeout(TOOL_TIMEOUT, connect=_CONNECT_TIMEOUT),
        )
    except httpx.TimeoutException as exc:
        raise GatewayError(
            504, f"节点 {node} 调用工具 {tool!r} 超时（>{TOOL_TIMEOUT:.0f}s）"
        ) from exc
    except httpx.HTTPError as exc:
        raise GatewayError(
            502, f"节点 {node} 不可达：{type(exc).__name__}: {exc}"
        ) from exc

    if resp.status_code == 404:
        # 目录里还有、节点上没了：目录过期。提示刷新，下一轮 refresh 后自愈
        raise GatewayError(
            404, f"节点 {node} 上已没有工具 {tool!r}（目录过期？请 POST /api/tools/refresh）"
        )
    if resp.status_code >= 500:
        raise GatewayError(502, f"节点 {node} 内部错误（HTTP {resp.status_code}）")
    if resp.status_code != 200:
        raise GatewayError(502, f"节点 {node} 异常响应（HTTP {resp.status_code}）")

    return InvokeResponse.model_validate(resp.json())


# ---------------------------------------------------------------- 会话
# Session / SessionStore 见 llm_node/sessions.py（SQLite 持久化）。

def _history_view(history: list[Any]) -> list[dict[str, Any]]:
    """把会话历史压成前端可重放的视图。

    工具结果消息（ToolMessage）不单列——前端重放时 AI 消息上带一条
    「调用了 xx 工具」的小字即可，完整 JSON 只在实时对话里有意义。
    """
    out: list[dict[str, Any]] = []
    for m in history:
        if isinstance(m, HumanMessage):
            out.append({
                "role": "user",
                "text": m.content if isinstance(m.content, str) else "",
            })
        elif isinstance(m, AIMessage):
            item: dict[str, Any] = {
                "role": "assistant",
                "text": m.content if isinstance(m.content, str) else "",
            }
            calls = getattr(m, "tool_calls", None) or []
            if calls:
                item["tool_calls"] = [
                    {"name": c.get("name", ""), "args": c.get("args", {})}
                    for c in calls
                ]
            out.append(item)
    return out

# ---------------------------------------------------------------- 请求/响应模型

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 id，前端生成并保持")
    message: str = Field(min_length=1)
    image: str | None = Field(
        default=None, description="base64，可带 data: 前缀；作为本会话当前图片"
    )
    stream: bool = Field(
        default=False,
        description="true 时以 SSE 流式返回（data: {type: delta|tool|done|error}）；"
        "默认 false 保持一次性的 JSON 响应",
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    history_len: int = Field(
        default=0, description="本轮结束后会话历史条数（已按窗口裁剪）"
    )
    history_max: int = Field(
        default=0, description="历史窗口上限（MAX_HISTORY_MESSAGES），0 = 未提供"
    )


def _sse(payload: dict[str, Any]) -> str:
    """把一条事件编码成 SSE 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_events(
    http: httpx.AsyncClient,
    catalog: ToolCatalog,
    session: Any,
    req: ChatRequest,
) -> AsyncIterator[str]:
    """流式对话的 SSE 事件源。

    事件类型：`delta`（文本增量）/ `tool`（一条工具记录）/ `done`（结束，
    带完整 reply）/ `error`。

    注意：异常发生在响应头已经发出之后，没法再改 HTTP 状态码（504/502），
    所以流里的失败一律以 `error` 事件表达——这是 SSE 的固有限制。
    """
    def invoke(tool: str, image: str | None, params: dict[str, Any]) -> Any:
        return invoke_tool(http, catalog, tool, image, params)

    if mock_enabled():
        reply, records = await agent.mock_chat(req.message, session.image, catalog.specs())
        yield _sse({"type": "delta", "text": reply})
        for r in records:
            yield _sse({"type": "tool", **r})
        yield _sse({"type": "done", "reply": reply, "tool_calls": records})
        return

    # 只在**这一轮上传了新图**时预取。会话图会跨轮留存，若按 session.image
    # 判断，用户之后随便说句"谢谢你"都会把 detect/classify 重跑一遍，
    # 而且观察被重新注入会把模型带偏成继续描述图片。
    # 追问轮（本轮没传新图但会话有图）给模型一句提示：图片工具仍可用
    # （工具自动取会话图），否则小模型会误以为"没图"而让用户重传。
    observations, prefetch = None, []
    if req.image and session.image:
        observations, prefetch = await _prefetch_observations(http, catalog, session.image)
    elif session.image:
        observations = _FOLLOWUP_IMAGE_HINT
    collected: list[dict[str, Any]] = list(prefetch)
    for r in prefetch:
        yield _sse({"type": "tool", **r})

    try:
        async for ev in agent.stream_chat(
            session.history,
            req.message,
            req.image,  # 只在本轮附图；追问轮靠历史里那张图
            agent.build_tools(catalog.specs(), invoke),
            observations=observations,
            tool_image=session.image,  # 追问轮模型仍可能调工具，得给着图
        ):
            if ev["type"] == "final":
                session.history = agent.trim_history(
                    [
                        *session.history,
                        HumanMessage(content=req.message),
                        *ev["messages"],
                    ],
                    MAX_HISTORY_MESSAGES,
                )
                yield _sse({
                    "type": "done",
                    "reply": ev["reply"],
                    "tool_calls": collected,
                    "history_len": len(session.history),
                    "history_max": MAX_HISTORY_MESSAGES,
                })
                return
            if ev["type"] == "ping":
                # SSE 注释帧：只为保活，前端解析器（只认 data: 行）自动忽略
                yield ": keepalive\n\n"
                continue
            if ev["type"] == "tool":
                collected.append({k: v for k, v in ev.items() if k != "type"})
            yield _sse(ev)
    except GraphRecursionError:
        yield _sse({
            "type": "error",
            "message": f"工具调用轮数超过上限（{agent.MAX_TOOL_ROUNDS}），请简化问题或稍后再试。",
        })
    except APITimeoutError as exc:
        yield _sse({"type": "error", "message": f"LLM 响应超时：{exc}"})
    except APIConnectionError as exc:
        yield _sse({
            "type": "error",
            "message": f"LLM 服务不可达（{llm.vllm_base_url()}），请先启动 vLLM：{exc}",
        })
    except Exception as exc:  # noqa: BLE001
        # 兜底：响应头早已发出，异常若直接抛出去会掐断连接，浏览器只看到一句
        # 无从下手的 "Load failed"。转成 error 事件，前端能显示原因，日志留全栈。
        logging.getLogger("llm_node.gateway").exception("流式对话失败")
        yield _sse({
            "type": "error",
            "message": f"生成中断：{type(exc).__name__}: {exc}",
        })


# ---------------------------------------------------------------- 应用工厂

def build_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient()
        try:
            try:
                await refresh_catalog(app.state.http, app.state.catalog)
            except DuplicateToolError as exc:
                # 启动遇重名不阻断：网关照常起，/api/health 能用，重定义留给人工
                logging.getLogger("llm_node.gateway").error("启动工具发现失败：%s", exc)
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(title="mm-agent gateway", version="1.0", lifespan=lifespan)
    # 前端（各成员自己的展示页）与网关不同源，浏览器直连要放行跨域。
    # 加在最外层，OPTIONS 预检由中间件直接应答。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.catalog = ToolCatalog()
    app.state.last_refresh: dict[str, Any] = {}
    app.state.sessions = SessionStore(DATA_DIR / "sessions.db")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """浏览器测试台：上传图片 + 对话 + 工具调用可视化。"""
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        """聚合健康状态（文档 §4.3：联调先看它）。始终 200，逐项给 ok。"""
        http: httpx.AsyncClient = request.app.state.http
        nodes: dict[str, Any] = {}
        for node in VISION_NODES:
            started = time.perf_counter()
            try:
                resp = await http.get(
                    f"{node_url(node)}/health", timeout=_HEALTH_TIMEOUT
                )
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                body = resp.json() if resp.status_code == 200 else {}
                nodes[node] = {
                    "ok": resp.status_code == 200,
                    "status": body.get("status"),
                    "tools": body.get("tools", []),
                    "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}",
                    "elapsed_ms": elapsed,
                }
            except Exception as exc:  # noqa: BLE001
                nodes[node] = {
                    "ok": False,
                    "status": None,
                    "tools": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }

        vllm = await llm.probe(http)
        catalog: ToolCatalog = request.app.state.catalog
        all_ok = all(n["ok"] for n in nodes.values()) and vllm["ok"]
        return {
            "gateway": {"status": "ok", "tools": catalog.names()},
            "nodes": nodes,
            "vllm": vllm,
            "all_ok": all_ok,
        }

    @app.get("/api/tools")
    async def tools(request: Request) -> dict[str, Any]:
        """聚合后的工具清单（Agent 靠它构建工具）。"""
        catalog: ToolCatalog = request.app.state.catalog
        return {
            "gateway": "llm",
            "tools": [s.model_dump() for s in catalog.specs()],
            "last_refresh": request.app.state.last_refresh,
        }

    @app.post("/api/tools/refresh")
    async def refresh(request: Request) -> dict[str, Any]:
        """重新发现工具（节点加了工具后调，文档 §5）。"""
        try:
            report = await refresh_catalog(request.app.state.http, request.app.state.catalog)
        except DuplicateToolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        request.app.state.last_refresh = report
        catalog: ToolCatalog = request.app.state.catalog
        return {
            "ok": all(n["ok"] for n in report.values()),
            "nodes": report,
            "tools": catalog.names(),
        }

    @app.post("/api/invoke", response_model=InvokeResponse)
    async def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
        """统一工具调用入口（tool 可来自任意节点）。"""
        try:
            return await invoke_tool(
                request.app.state.http, request.app.state.catalog,
                req.tool, req.image, req.params,
            )
        except GatewayError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/sessions")
    async def list_sessions(request: Request) -> dict[str, Any]:
        """历史会话列表（新→旧），前端会话选择器用。"""
        return {"sessions": request.app.state.sessions.list_sessions()}

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str, request: Request) -> dict[str, Any]:
        """单个会话详情：消息历史视图 + 会话当前图片（前端切换会话时重放）。"""
        store = request.app.state.sessions
        if not store.exists(session_id):
            raise HTTPException(status_code=404, detail=f"会话 {session_id!r} 不存在")
        session = store.get(session_id)
        return {
            "session_id": session_id,
            "messages": _history_view(session.history),
            "image": session.image,
            "history_len": len(session.history),
            "history_max": MAX_HISTORY_MESSAGES,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> Any:
        """对话主入口。可带图；同一 session_id 多轮续聊。

        `stream=true` 时返回 SSE（见 `_stream_events`），否则返回一次性 JSON。
        """
        http: httpx.AsyncClient = request.app.state.http
        catalog: ToolCatalog = request.app.state.catalog
        session = request.app.state.sessions.get(req.session_id)
        if req.image:
            session.image = req.image

        if req.stream:
            return StreamingResponse(
                _stream_events(http, catalog, session, req),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        specs = catalog.specs()

        if mock_enabled():
            reply, records = await agent.mock_chat(req.message, session.image, specs)
            request.app.state.sessions.save(req.session_id, session)
            return ChatResponse(
                reply=reply, tool_calls=records,
                history_len=len(session.history), history_max=MAX_HISTORY_MESSAGES,
            )

        # 只在**这一轮上传了新图**时预取（与流式路径同一判断，理由见 _stream_events）；
        # 追问轮给模型一句提示：会话图仍在，图片工具可直接用（见 _FOLLOWUP_IMAGE_HINT）
        observations, prefetch_records = None, []
        if req.image and session.image:
            observations, prefetch_records = await _prefetch_observations(
                http, catalog, session.image
            )
        elif session.image:
            observations = _FOLLOWUP_IMAGE_HINT

        def invoke(tool: str, image: str | None, params: dict[str, Any]) -> Any:
            return invoke_tool(http, catalog, tool, image, params)

        try:
            reply, records, new_msgs = await agent.run_chat(
                session.history, req.message, req.image,
                agent.build_tools(specs, invoke),
                observations=observations,
                tool_image=session.image,
            )
        except GraphRecursionError:
            return ChatResponse(
                reply=f"[错误] 工具调用轮数超过上限（{agent.MAX_TOOL_ROUNDS}），"
                      "请简化问题或稍后再试。",
                tool_calls=[],
            )
        except APITimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"LLM 响应超时：{exc}") from exc
        except APIConnectionError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"LLM 服务不可达（{llm.vllm_base_url()}），请先启动 vLLM：{exc}",
            ) from exc
        except APIStatusError as exc:
            # vLLM 有响应但拒绝了请求：404 模型名对不上 / 400 参数或图片超长 / 429 过载
            raise HTTPException(
                status_code=502,
                detail=f"vLLM 返回 {exc.status_code}：{exc.message}"
                       "（404 时先核对 VLLM_MODEL_NAME 是否与 /v1/models 里的 id 一致）",
            ) from exc

        # 历史拼装：用户消息也进历史（P0，只存文本），并按调用对对齐裁剪（P1）
        session.history = agent.trim_history(
            [
                *session.history,
                HumanMessage(content=req.message),
                *new_msgs,
            ],
            MAX_HISTORY_MESSAGES,
        )
        request.app.state.sessions.save(req.session_id, session)
        return ChatResponse(
            reply=reply, tool_calls=prefetch_records + records,
            history_len=len(session.history), history_max=MAX_HISTORY_MESSAGES,
        )

    return app


app = build_app()
