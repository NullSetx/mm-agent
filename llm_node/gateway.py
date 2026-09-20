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

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field

from common.config import TOOL_TIMEOUT, mock_enabled
from common.schemas import InvokeRequest, InvokeResponse, ToolSpec, ToolList
from llm_node import agent, llm

#: 接进网关的视觉节点。新增节点：config.HOSTS/PORTS 加键 + 这里加名字
VISION_NODES: tuple[str, ...] = ("vision-fast", "vision-heavy")

_DISCOVER_TIMEOUT = 5.0  # 发现 /health 探测的兜底超时（秒）
_HEALTH_TIMEOUT = 3.0
_CONNECT_TIMEOUT = 5.0

#: 会话历史保留的最大消息条数（含工具消息）
MAX_HISTORY_MESSAGES = 12


def node_url(node: str) -> str:
    """拼视觉节点基址，如 http://192.168.1.101:8101。

    不用 common.config.base_url：它对 HOSTS 有而 PORTS 没有的键会 KeyError、
    未知键会静默错路由到 vision-heavy（common/config.py 现成问题，已报群里）。
    """
    from common.config import HOSTS, PORTS  # 调用时取值，测试改环境变量才生效

    return f"http://{HOSTS[node]}:{PORTS[node]}"


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

@dataclass
class Session:
    """一个会话 = 消息历史 + 当前图片（每次传图覆盖，最近一张为准）。"""

    history: list[Any] = field(default_factory=list)
    image: str | None = None


class SessionStore:
    """内存会话表。进程重启即失效——文档未要求持久化，先不做。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        return self._sessions.setdefault(session_id, Session())


# ---------------------------------------------------------------- 请求/响应模型

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 id，前端生成并保持")
    message: str = Field(min_length=1)
    image: str | None = Field(
        default=None, description="base64，可带 data: 前缀；作为本会话当前图片"
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


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
    app.state.catalog = ToolCatalog()
    app.state.last_refresh: dict[str, Any] = {}
    app.state.sessions = SessionStore()

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

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        """对话主入口。可带图；同一 session_id 多轮续聊。"""
        http: httpx.AsyncClient = request.app.state.http
        catalog: ToolCatalog = request.app.state.catalog
        session = request.app.state.sessions.get(req.session_id)
        if req.image:
            session.image = req.image
        specs = catalog.specs()

        if mock_enabled():
            reply, records = await agent.mock_chat(req.message, session.image, specs)
            return ChatResponse(reply=reply, tool_calls=records)

        def invoke(tool: str, image: str | None, params: dict[str, Any]) -> Any:
            return invoke_tool(http, catalog, tool, image, params)

        try:
            reply, records, new_msgs = await agent.run_chat(
                session.history, req.message, session.image,
                agent.build_tools(specs, invoke),
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

        session.history = (session.history + list(new_msgs))[-MAX_HISTORY_MESSAGES:]
        return ChatResponse(reply=reply, tool_calls=records)

    return app


app = build_app()
