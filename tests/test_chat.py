"""`/api/chat` 端点测试：mock 模式端到端、真实循环端到端（假 LLM）、会话与图片。"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from llm_node import gateway as gw, llm
from tests.conftest import make_toy_node

FAST = gw.node_url("vision-fast")


class FakeAgentModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


def _tiny_png_base64() -> str:
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


@pytest.mark.anyio
async def test_mock_chat_end_to_end(gateway, router, monkeypatch):
    """NODE_MOCK=1：无 LLM、无下游也能拿到结构合法的响应（文档 §6.2）。"""
    monkeypatch.setenv("NODE_MOCK", "1")
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")

    resp = await gateway.post(
        "/api/chat", json={"session_id": "s1", "message": "你好"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "mock" in body["reply"]
    assert body["tool_calls"] == []

    # 带图 → mock 记录一次对第一个带图工具的调用；图片存进会话
    resp = await gateway.post(
        "/api/chat",
        json={"session_id": "s1", "message": "看看", "image": _tiny_png_base64()},
    )
    body = resp.json()
    assert body["tool_calls"] == [{"tool": "echo", "ok": True, "result": {"mock": True}}]
    session = router["_app"].state.sessions.get("s1")
    assert session.image == _tiny_png_base64()


@pytest.mark.anyio
async def test_chat_real_loop_end_to_end(gateway, router, monkeypatch):
    """不带 NODE_MOCK 走真实 Agent 路径：假 LLM 两次回复，真工具真执行。"""
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")

    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "echo", "args": {}, "id": "c1", "type": "tool_call"},
            ]),
            AIMessage(content="回答完毕"),
        ]),
    )

    resp = await gateway.post(
        "/api/chat", json={"session_id": "s2", "message": "测试一下"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "回答完毕"
    assert body["tool_calls"] == [
        {"tool": "echo", "ok": True, "result": {"pong": True}, "error": None}
    ]

    # 会话历史包含用户消息（P0：多轮上下文不再丢用户的话）+ 工具消息
    session = router["_app"].state.sessions.get("s2")
    assert len(session.history) == 4
    assert session.history[0].type == "human"
    assert session.history[0].content == "测试一下"


@pytest.mark.anyio
async def test_chat_trims_history(gateway, router, monkeypatch):
    """历史超过上限时只保留最近 MAX_HISTORY_MESSAGES 条。"""
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[AIMessage(content="好")]),
    )

    # 每轮历史 +2 条，轮数按当前窗口值算，必然触发裁剪（窗口被调大也成立）
    rounds = gw.MAX_HISTORY_MESSAGES // 2 + 2
    for i in range(rounds):
        resp = await gateway.post(
            "/api/chat", json={"session_id": "s3", "message": f"第{i}轮"}
        )
        assert resp.status_code == 200

    # 前端画"记忆分界线"依赖这两个字段（最后一轮后窗口已满）
    body = resp.json()
    assert body["history_len"] == gw.MAX_HISTORY_MESSAGES
    assert body["history_max"] == gw.MAX_HISTORY_MESSAGES

    session = router["_app"].state.sessions.get("s3")
    assert len(session.history) == gw.MAX_HISTORY_MESSAGES
    assert session.history[-1].content == "好"


@pytest.mark.anyio
async def test_chat_llm_down_maps_502(gateway, router, monkeypatch):
    """vLLM 不在线时，/api/chat 返回 502 并提示先启动 vLLM。"""
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")
    # 指向一个必然无人监听的端口，不依赖"真 vLLM 恰好没开"
    monkeypatch.setattr(
        llm, "vllm_base_url", lambda: "http://127.0.0.1:59999"
    )
    resp = await gateway.post(
        "/api/chat", json={"session_id": "s4", "message": "你好"}
    )
    assert resp.status_code == 502
    assert "vLLM" in resp.json()["detail"]


@pytest.mark.anyio
async def test_chat_model_mismatch_maps_502(gateway, router, monkeypatch):
    """vLLM 拒绝请求（如模型名对不上）→ 502 可读提示，而不是裸 500。"""
    import httpx as _httpx
    import openai as _openai

    def raise_404(**kw):  # noqa: ANN001, ANN003
        request = _httpx.Request("POST", "http://vllm.test/v1/chat/completions")
        response = _httpx.Response(404, request=request, json={"error": {"message": "no"}})
        raise _openai.APIStatusError(
            "Error code: 404 - The model `x` does not exist.",
            response=response, body=None,
        )

    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(llm, "build_chat_model", raise_404)

    resp = await gateway.post(
        "/api/chat", json={"session_id": "s5", "message": "你好"}
    )
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "404" in detail and "VLLM_MODEL_NAME" in detail


@pytest.mark.anyio
async def test_session_list_and_detail_endpoints(gateway, router, monkeypatch):
    """GET /api/sessions 列表 + /{id} 详情：切换历史会话的数据源。"""
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "echo", "args": {}, "id": "c1", "type": "tool_call"},
            ]),
            AIMessage(content="回答完毕"),
        ]),
    )
    await gateway.post("/api/chat", json={"session_id": "hist-1", "message": "测试一下"})

    listing = (await gateway.get("/api/sessions")).json()
    ids = [s["session_id"] for s in listing["sessions"]]
    assert "hist-1" in ids

    detail = (await gateway.get("/api/sessions/hist-1")).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "assistant"]
    assert detail["messages"][0]["text"] == "测试一下"
    assert detail["messages"][1]["tool_calls"] == [{"name": "echo", "args": {}}]
    assert detail["messages"][2]["text"] == "回答完毕"

    # 未落库的会话 404
    resp = await gateway.get("/api/sessions/nope")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_chat_stream_sse_frames(gateway, router, monkeypatch):
    """stream=true 走 SSE：text/event-stream，delta/tool/done 帧齐全。

    （回归保护：这条链路曾在工作区改动中被整体误删而测试无感。）
    """
    import json as _json

    monkeypatch.setenv("NODE_MOCK", "1")
    resp = await gateway.post(
        "/api/chat",
        json={"session_id": "s6", "message": "你好", "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = []
    for frame in resp.text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                events.append(_json.loads(line[6:]))
    types = [e["type"] for e in events]
    assert types[0] == "delta"
    assert types[-1] == "done"
    done = events[-1]
    assert "mock" in done["reply"]
    assert done["tool_calls"] == []
