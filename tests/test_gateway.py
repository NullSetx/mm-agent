"""网关契约测试：工具发现 / 重名 / 转发错误映射 404-502-504 / 聚合健康。"""

from __future__ import annotations

import pytest

from llm_node import gateway as gw
from tests.conftest import TimeoutTransport, fake_vllm_transport, make_toy_node

FAST = gw.node_url("vision-fast")
HEAVY = gw.node_url("vision-heavy")


def echo_tools() -> list[dict]:
    return [
        {"name": "echo", "description": "回声", "needs_image": False, "fn": lambda: {"pong": True}},
    ]


def both_nodes(router) -> None:
    """挂上两个各含一个工具的节点（工具名不同）。"""
    router[FAST] = make_toy_node("vision-fast", echo_tools())
    router[HEAVY] = make_toy_node(
        "vision-heavy",
        [{"name": "slowpoke", "description": "带图工具", "needs_image": True,
          "params": {"conf": 0.25},
          "fn": lambda image, conf=0.25: {"conf": conf}}],
    )


@pytest.mark.anyio
async def test_refresh_aggregates_and_invoke_forwards(gateway, router):
    both_nodes(router)
    resp = await gateway.post("/api/tools/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["tools"] == ["echo", "slowpoke"]

    listing = (await gateway.get("/api/tools")).json()
    spec_names = {t["name"] for t in listing["tools"]}
    assert spec_names == {"echo", "slowpoke"}
    # ToolSpec 原样透传，params 默认值保留
    slowpoke = next(t for t in listing["tools"] if t["name"] == "slowpoke")
    assert slowpoke["params"] == {"conf": 0.25}
    assert slowpoke["needs_image"] is True

    # invoke 透传到正确节点
    resp = await gateway.post("/api/invoke", json={"tool": "echo"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["result"] == {"pong": True}


@pytest.mark.anyio
async def test_refresh_duplicate_returns_409_and_keeps_old_catalog(gateway, router):
    router[FAST] = make_toy_node("vision-fast", echo_tools())
    assert (await gateway.post("/api/tools/refresh")).status_code == 200

    # heavy 也登记 echo → 重名 → 409，且目录保留刷新前的内容
    router[HEAVY] = make_toy_node("vision-heavy", echo_tools())
    resp = await gateway.post("/api/tools/refresh")
    assert resp.status_code == 409
    assert "echo" in resp.json()["detail"]

    listing = (await gateway.get("/api/tools")).json()
    assert [t["name"] for t in listing["tools"]] == ["echo"]
    assert listing["tools"][0]  # 仍可正常调用
    assert (await gateway.post("/api/invoke", json={"tool": "echo"})).json()["ok"] is True


@pytest.mark.anyio
async def test_refresh_node_down_drops_its_tools(gateway, router):
    both_nodes(router)
    await gateway.post("/api/tools/refresh")

    # heavy 下线（从路由表摘除 = 失联）
    del router[HEAVY]
    body = (await gateway.post("/api/tools/refresh")).json()
    assert body["ok"] is False
    assert body["nodes"]["vision-heavy"]["ok"] is False
    assert body["nodes"]["vision-heavy"]["error"] is not None
    assert body["tools"] == ["echo"]  # 失联节点的工具被摘除


@pytest.mark.anyio
async def test_invoke_unknown_tool_404(gateway, router):
    both_nodes(router)
    await gateway.post("/api/tools/refresh")
    resp = await gateway.post("/api/invoke", json={"tool": "nope"})
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


@pytest.mark.anyio
async def test_invoke_node_down_502(gateway, router):
    both_nodes(router)
    await gateway.post("/api/tools/refresh")
    del router[HEAVY]  # 目录还没刷新，节点已经挂了
    resp = await gateway.post("/api/invoke", json={"tool": "slowpoke", "image": "x"})
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_invoke_downstream_timeout_504(gateway, router):
    """先发现成功，再把节点传输层换成"挂起不响应"——验证超时映射 504。
    （ASGITransport 不执行客户端超时，所以用抛 ReadTimeout 的假传输层。）"""
    router[FAST] = make_toy_node("vision-fast", echo_tools())
    await gateway.post("/api/tools/refresh")
    router[FAST] = TimeoutTransport()
    resp = await gateway.post("/api/invoke", json={"tool": "echo"})
    assert resp.status_code == 504


@pytest.mark.anyio
async def test_invoke_tool_failure_is_200_ok_false(gateway, router):
    def broken() -> dict:
        raise RuntimeError("故意炸的")

    router[FAST] = make_toy_node(
        "vision-fast",
        [{"name": "broken", "description": "必炸", "needs_image": False, "fn": broken}],
    )
    await gateway.post("/api/tools/refresh")
    resp = await gateway.post("/api/invoke", json={"tool": "broken"})
    assert resp.status_code == 200  # 工具内部错误不拖垮网关
    body = resp.json()
    assert body["ok"] is False
    assert "RuntimeError" in body["error"]


@pytest.mark.anyio
async def test_health_aggregates_nodes_and_vllm(gateway, router):
    both_nodes(router)
    # 都在线 + vLLM 在线 → all_ok
    router["http://127.0.0.1:8001"] = fake_vllm_transport()
    body = (await gateway.get("/api/health")).json()
    assert body["all_ok"] is True
    assert body["nodes"]["vision-fast"]["ok"] is True
    assert body["vllm"]["ok"] is True
    assert body["vllm"]["models"] == ["fake-model"]

    # heavy 掉线 → all_ok 变 False，但整体仍 200
    del router[HEAVY]
    body = (await gateway.get("/api/health")).json()
    assert body["all_ok"] is False
    assert body["nodes"]["vision-heavy"]["ok"] is False


@pytest.mark.anyio
async def test_stale_catalog_returns_404_with_refresh_hint(gateway, router):
    router[FAST] = make_toy_node("vision-fast", echo_tools())
    await gateway.post("/api/tools/refresh")
    # 节点端工具消失（换成了没带 echo 的版本），网关目录未刷新
    router[FAST] = make_toy_node("vision-fast", [])
    resp = await gateway.post("/api/invoke", json={"tool": "echo"})
    assert resp.status_code == 404
    assert "refresh" in resp.json()["detail"]
