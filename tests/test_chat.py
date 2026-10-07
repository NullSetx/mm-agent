"""`/api/chat` 端点测试：mock 模式端到端、真实循环端到端（假 LLM）、会话与图片。"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from llm_node import agent as agent_mod, gateway as gw, llm
from llm_node.sessions import Session
from tests.conftest import make_toy_node

FAST = gw.node_url("vision-fast")
HEAVY = gw.node_url("vision-heavy")


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
async def test_followup_round_gets_image_hint(gateway, router, monkeypatch):
    """追问轮（本轮没传新图、会话有图）：模型收到"会话图仍可用"的提示。

    回归背景：模型把"本轮没传图"误读成"没图可用"，让用户重传会话里
    明明已有的图片。
    """
    captured: dict = {}

    class RecordingModel(FakeAgentModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    router[FAST] = make_toy_node("vision-fast", [
        {"name": "ocr", "description": "读字", "needs_image": True,
         "params": {}, "fn": lambda image: {"full_text": ""}},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: RecordingModel(responses=[AIMessage(content="好")]),
    )

    img = _tiny_png_base64()
    await gateway.post("/api/chat",
                       json={"session_id": "s7", "message": "看看", "image": img})
    await gateway.post("/api/chat", json={"session_id": "s7", "message": "提取文字"})

    human = [m for m in captured["messages"]
             if getattr(m, "type", "") == "human"][-1]
    assert "会话中已有一张此前上传的图片" in str(human.content)


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
async def test_session_delete_endpoint(gateway, router, monkeypatch):
    """DELETE /api/sessions/{id}：成功 200 + 列表消失；不存在 404。"""
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[AIMessage(content="好")]),
    )
    await gateway.post("/api/chat", json={"session_id": "del-me", "message": "嗨"})

    resp = await gateway.delete("/api/sessions/del-me")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "deleted": "del-me"}
    ids = [s["session_id"] for s in (await gateway.get("/api/sessions")).json()["sessions"]]
    assert "del-me" not in ids

    assert (await gateway.delete("/api/sessions/del-me")).status_code == 404


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


@pytest.mark.anyio
async def test_chat_stream_persists_session(gateway, router, monkeypatch):
    """stream=true 也必须把会话落盘，否则多轮就是每轮空会话。

    回归保护：流式分支在 _stream_events 里更新 session.history，但端点函数随
    `return StreamingResponse(...)` 就退出了，保存只能由响应体生成器自己做。
    漏了那步不会报错——SSE 的 done 事件照样带着 history_len，前端看着很正常，
    只有回到数据库才发现会话根本没写。
    """
    router[FAST] = make_toy_node("vision-fast", [
        {"name": "echo", "description": "回声", "needs_image": False,
         "fn": lambda: {"pong": True}},
    ])
    await gateway.post("/api/tools/refresh")

    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[AIMessage(content="回答完毕")]),
    )

    store = router["_app"].state.sessions
    resp = await gateway.post(
        "/api/chat",
        json={"session_id": "s7", "message": "测试一下", "stream": True},
    )
    assert resp.status_code == 200

    # SessionStore 没有内存缓存，get() 就是查 SQLite —— 读得到即已落盘
    session = store.get("s7")
    assert [m.content for m in session.history] == ["测试一下", "回答完毕"]

    # 第二轮若从空会话开始，落盘后只会有 2 条；有 4 条才说明上一轮真的被读回来了
    resp = await gateway.post(
        "/api/chat",
        json={"session_id": "s7", "message": "再问一次", "stream": True},
    )
    assert resp.status_code == 200
    session = store.get("s7")
    assert [m.content for m in session.history] == [
        "测试一下", "回答完毕", "再问一次", "回答完毕",
    ]


# ---------------------------------------------------------------- 附件分流

def test_split_attachment():
    """只看 data URI 前缀声明的 mime；无前缀沿用老语义当图片（A 的测试台和
    历史客户端发的都是裸 base64，这条不能改）。"""
    assert gw.split_attachment(None) == ("none", "", "")
    assert gw.split_attachment("   ") == ("none", "", "")

    assert gw.split_attachment("AAAA")[0] == "image"
    assert gw.split_attachment("data:image/png;base64,AAAA")[0] == "image"

    kind, mime, payload = gw.split_attachment("data:application/pdf;base64,JVBERi0=")
    assert (kind, mime, payload) == ("document", "application/pdf", "JVBERi0=")
    assert gw.split_attachment("data:text/x-python;base64,ZA==")[0] == "document"


def test_split_attachment_rejects_broken_data_uri():
    with pytest.raises(gw.GatewayError):
        gw.split_attachment("data:application/pdf;base64")


@pytest.mark.anyio
async def test_pdf_attachment_goes_through_read_document(gateway, router, monkeypatch):
    """PDF 走文档通道：交给 read_document 读成正文拼进消息。

    两个关键点：正文要随历史留存（否则追问「第 3 页说了什么」时内容已经不在
    上下文里），且**不能**存进 session.image（那是"当前图片"，工具会拿它当输入图，
    给 detect / ocr 一份 PDF 只会报"不是有效图片"）。
    """
    seen: list[dict] = []

    def fake_read_document(_data="", _mime="", _name=""):
        seen.append({"data": _data, "mime": _mime, "name": _name})
        return {"kind": "pdf", "text": "--- 第 1 页 ---\n文档正文在这里", "note": "共 1 页"}

    router[HEAVY] = make_toy_node("vision-heavy", [
        {"name": "read_document", "description": "读文件", "needs_image": False,
         "params": {"_data": "", "_mime": "", "_name": ""}, "fn": fake_read_document},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[AIMessage(content="读到了")]),
    )

    payload = base64.b64encode(b"%PDF-1.4 fake").decode()
    resp = await gateway.post("/api/chat", json={
        "session_id": "d1", "message": "这份 PDF 讲了什么？",
        "image": f"data:application/pdf;base64,{payload}",
    })
    assert resp.status_code == 200
    assert resp.json()["reply"] == "读到了"

    assert seen and seen[0]["data"] == payload
    assert seen[0]["mime"] == "application/pdf"

    session = router["_app"].state.sessions.get("d1")
    human = [m for m in session.history if m.type == "human"][0]
    assert "文档正文在这里" in human.content          # 正文进了历史
    assert "共 1 页" in human.content                 # 附注也带上
    assert session.image is None                     # 没被当成"当前图片"


@pytest.mark.anyio
async def test_image_attachment_keeps_old_path(gateway, router, monkeypatch):
    """图片仍走老路：不经过 read_document，session.image 存下来供工具用。"""
    seen: list[dict] = []

    def fake_read_document(_data="", _mime="", _name=""):
        seen.append({"data": _data})
        return {"kind": "text", "text": "", "note": ""}

    router[HEAVY] = make_toy_node("vision-heavy", [
        {"name": "read_document", "description": "读文件", "needs_image": False,
         "params": {"_data": "", "_mime": "", "_name": ""}, "fn": fake_read_document},
    ])
    await gateway.post("/api/tools/refresh")
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[AIMessage(content="看到了")]),
    )

    png = f"data:image/png;base64,{_tiny_png_base64()}"
    resp = await gateway.post("/api/chat", json={
        "session_id": "d2", "message": "这是什么？", "image": png,
    })
    assert resp.status_code == 200

    assert seen == []                                # 文档通道没被碰
    session = router["_app"].state.sessions.get("d2")
    assert session.image == png
    human = [m for m in session.history if m.type == "human"][0]
    assert human.content == "这是什么？"               # 图片不往消息里塞正文


# ---------------------------------------------------------------- 上下文预算

_seen_inputs: list[list] = []


class RecordingAgentModel(FakeAgentModel):
    """记录模型实际收到的消息，用来断言"发之前裁过"。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _seen_inputs.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.anyio
async def test_oversized_history_is_trimmed_before_sending(gateway, router, monkeypatch):
    """库里已经躺着的超长历史，也要在**发之前**裁掉。

    只在轮末裁挡不住这种情况：那批历史是上一次配置留下的，这一轮会原样发出去、
    被 vLLM 400 拒掉——实测就是这么炸的（报错还是"maximum context length is
    10240 tokens"这种用户完全无从下手的话）。
    """
    store = router["_app"].state.sessions
    store.save("big", Session(history=[HumanMessage(content="X" * 5000) for _ in range(4)]))

    # 把预算钉死，免得测试结果随 .env / 探到的 vLLM 窗口变化
    monkeypatch.setattr(gw, "history_char_budget", lambda: 6000)

    _seen_inputs.clear()
    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: RecordingAgentModel(responses=[AIMessage(content="好")]),
    )

    resp = await gateway.post("/api/chat", json={"session_id": "big", "message": "接着说"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "好"

    sent = _seen_inputs[-1]
    total = sum(agent_mod._msg_chars(m) for m in sent)
    assert total <= 6000, f"发出去了 {total} 字符，超过预算"
    # 最新那条是用户刚说的话，绝不能丢
    assert "接着说" in str(sent[-1].content)

    # 轮末也要裁：落库的历史总量回到预算内，下一轮不至于又超
    persisted = store.get("big").history
    kept = sum(agent_mod._msg_chars(m) for m in persisted)
    assert kept <= 6000
    assert kept < 20000, "轮末没裁，库里还是原始的超长历史"
