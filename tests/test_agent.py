"""Agent 层测试：动态工具 schema、错误兜底、mock 对话、对话循环（假 LLM + 真工具）。"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from common.schemas import InvokeResponse, ToolSpec
from llm_node import agent, gateway as gw
from tests.conftest import make_toy_node


class FakeAgentModel(FakeMessagesListChatModel):
    """按顺序回放消息的假 LLM。bind_tools 返回自身以兼容 create_agent。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


# ---------------------------------------------------------------- schema

def test_args_schema_types_from_defaults():
    spec = ToolSpec(
        name="demo", description="d",
        params={"conf": 0.25, "n": 3, "flag": True, "s": "x", "free": None},
    )
    schema = agent.args_schema(spec)
    assert schema is not None
    fields = schema.model_fields
    assert fields["conf"].annotation is float
    assert fields["n"].annotation is int
    assert fields["flag"].annotation is bool
    assert fields["s"].annotation is str
    assert fields["free"].default is None
    assert fields["conf"].default == 0.25


def test_args_schema_empty_params():
    spec = ToolSpec(name="t", description="d", params={})
    assert agent.args_schema(spec) is None


# ---------------------------------------------------------------- 工具兜底

@pytest.mark.anyio
async def test_tool_payload_reports_failure_not_raise():
    async def invoke(tool, image, params):
        raise gw.GatewayError(502, "节点不可达")

    tool = agent.build_tool(ToolSpec(name="x", description="d", needs_image=False), invoke)
    payload = json.loads(await tool.ainvoke({}))
    assert payload["ok"] is False
    assert "GatewayError" in payload["error"]


# ---------------------------------------------------------------- mock 对话

@pytest.mark.anyio
async def test_mock_chat_without_image():
    specs = [ToolSpec(name="echo", description="回声", needs_image=False)]
    reply, calls = await agent.mock_chat("你好", None, specs)
    assert "echo" in reply
    assert calls == []


@pytest.mark.anyio
async def test_mock_chat_with_image_records_first_image_tool():
    specs = [
        ToolSpec(name="ocr", description="读字", needs_image=True),
        ToolSpec(name="echo", description="回声", needs_image=False),
    ]
    reply, calls = await agent.mock_chat("看看", "aGVsbG8=", specs)
    assert calls == [{"tool": "ocr", "ok": True, "result": {"mock": True}}]
    assert "ocr" in reply


# ---------------------------------------------------------------- 对话循环

def _tiny_png_base64() -> str:
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


@pytest.mark.anyio
async def test_run_chat_calls_tool_and_injects_image(gateway, router):
    """假 LLM 发起一次带参工具调用：图片由 ContextVar 注入而不是 LLM 传参。"""
    captured: dict = {}

    def needs_img(image, conf=0.25):
        captured["shape"] = list(image.shape)
        captured["conf"] = conf
        return {"conf": conf}

    router[gw.node_url("vision-fast")] = make_toy_node(
        "vision-fast",
        [{"name": "needs_img", "description": "带图", "needs_image": True,
          "params": {"conf": 0.25}, "fn": needs_img}],
    )
    catalog = router["_app"].state.catalog
    from llm_node.gateway import refresh_catalog

    await refresh_catalog(router["_app"].state.http, catalog)

    async def invoke(tool, image, params):
        return await gw.invoke_tool(router["_app"].state.http, catalog, tool, image, params)

    model = FakeAgentModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "needs_img", "args": {"conf": "0.5"}, "id": "c1", "type": "tool_call"},
        ]),
        AIMessage(content="图是黑的"),
    ])
    image = _tiny_png_base64()
    reply, records, new_msgs = await agent.run_chat(
        history=[], message="图里是什么", image=image,
        tools=agent.build_tools(catalog.specs(), invoke), chat_model=model,
    )

    assert reply == "图是黑的"
    assert captured["shape"] == [4, 6, 3]      # base64 被解码成真实图片传给了工具
    assert captured["conf"] == 0.5             # LLM 传的字符串被 args schema 强转
    assert len(records) == 1
    assert records[0]["tool"] == "needs_img"
    assert records[0]["ok"] is True
    assert records[0]["result"] == {"conf": 0.5}
    assert len(new_msgs) == 3                  # tool_call 消息 + 工具回填 + 最终回答


@pytest.mark.anyio
async def test_run_chat_tool_error_feeds_back_to_llm(gateway, router):
    """工具失败时错误进入对话，LLM（这里用回放消息模拟）仍能给出最终回答。"""
    router[gw.node_url("vision-fast")] = make_toy_node(
        "vision-fast",
        [{"name": "echo", "description": "回声", "needs_image": False,
          "fn": lambda: {"pong": True}}],
    )
    catalog = router["_app"].state.catalog
    from llm_node.gateway import refresh_catalog

    await refresh_catalog(router["_app"].state.http, catalog)

    async def invoke(tool, image, params):
        raise gw.GatewayError(502, "节点不可达")

    model = FakeAgentModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {}, "id": "c1", "type": "tool_call"},
        ]),
        AIMessage(content="工具挂了，抱歉"),
    ])
    reply, records, _ = await agent.run_chat(
        history=[], message="你好", image=None,
        tools=agent.build_tools(catalog.specs(), invoke), chat_model=model,
    )
    assert reply == "工具挂了，抱歉"
    assert records[0]["ok"] is False
    assert "GatewayError" in records[0]["error"]
