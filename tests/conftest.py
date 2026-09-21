"""llm_node 测试公共设施。

思路：下游视觉节点用 common.node.build_app 起真实契约的 ASGI app，
网关的 httpx 客户端换成一个按 origin 路由的假传输层——请求不落 socket，
但路径、状态码、错误语义全部真实。映射表缺某个 origin 就等于那个节点失联。
"""

from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest

from common.node import build_app
from common.registry import ToolRegistry
from llm_node import gateway as gw
from llm_node.sessions import SessionStore


class OriginRouter(httpx.AsyncBaseTransport):
    """按 origin（http://host:port）把请求转发到对应 ASGI app。

    没登记的 origin 视为下游不可达，直接抛 ConnectError（→ 网关映射 502）。
    """

    def __init__(self, mapping: dict[str, httpx.AsyncBaseTransport]):
        self.mapping = mapping

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        origin = f"http://{request.url.host}:{request.url.port}"
        transport = self.mapping.get(origin)
        if transport is None:
            raise httpx.ConnectError(f"测试路由表里没有 {origin}", request=request)
        return await transport.handle_async_request(request)


class TimeoutTransport(httpx.AsyncBaseTransport):
    """模拟下游超时（→ 网关映射 504）。"""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)


def make_toy_node(
    name: str,
    tools: list[dict[str, Any]],
    health_detail: dict[str, Any] | None = None,
) -> httpx.AsyncBaseTransport:
    """起一个符合 common 契约的玩具节点。

    tools 每项：{name, description, needs_image, params, fn}
    """
    reg = ToolRegistry()
    for t in tools:
        fn: Callable[..., Any] = t["fn"]
        reg.register(
            name=t["name"],
            description=t["description"],
            needs_image=t.get("needs_image", False),
            params=t.get("params"),
        )(fn)
    return httpx.ASGITransport(app=build_app(name, registry=reg, health_hook=lambda: health_detail or {}))


def fake_vllm_transport() -> httpx.AsyncBaseTransport:
    """假 vLLM：只回答 /v1/models，供 /api/health 探活。"""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "fake-model"}]}

    return httpx.ASGITransport(app=app)


@pytest.fixture
def router() -> dict[str, httpx.AsyncBaseTransport]:
    """origin -> transport 映射。测试里可以随时增删来模拟节点上/下线。"""
    return {}


@pytest.fixture
def gateway(router: dict[str, httpx.AsyncBaseTransport], monkeypatch, tmp_path):
    """带路由传输层的网关客户端。lifespan 不跑，http 客户端与会话存储由这里注入。

    会话存储用 tmp 下的真实 SQLite（隔离 + 走真实存储路径）。
    """
    monkeypatch.delenv("NODE_MOCK", raising=False)
    app = gw.build_app()
    app.state.sessions = SessionStore(tmp_path / "test_sessions.db")
    app.state.http = httpx.AsyncClient(transport=OriginRouter(router))
    router["_app"] = app  # 供测试检查 catalog / session 状态
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
