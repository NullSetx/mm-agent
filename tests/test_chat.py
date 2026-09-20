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

    # 会话历史被回填（tool_call 消息 + 工具回填 + 最终回答）
    session = router["_app"].state.sessions.get("s2")
    assert len(session.history) == 3


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

    for i in range(15):  # 每轮历史 +1 条，15 轮必然触发裁剪
        resp = await gateway.post(
            "/api/chat", json={"session_id": "s3", "message": f"第{i}轮"}
        )
        assert resp.status_code == 200

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
