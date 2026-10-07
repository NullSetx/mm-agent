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
from langchain_core.messages import HumanMessage, ToolMessage
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
    """无参数工具给显式空 schema，防止 LangChain 推断出误导性的 kwargs 属性。"""
    spec = ToolSpec(name="t", description="d", params={})
    schema = agent.args_schema(spec)
    assert schema is not None
    assert schema.model_fields == {}
    assert schema.model_json_schema()["properties"] == {}


def test_args_schema_hides_internal_params():
    """下划线开头的参数是内部参数，不能进 LLM 的 schema。

    它们是网关注入的（如 read_document 的文件内容），模型既拿不到也不该编。
    实测 4B 会照着自己臆造的值去调、拿到报错后**照着报错回答**，把已经放进
    上下文的正文全无视掉——所以这层过滤是必需的，不是洁癖。
    """
    spec = ToolSpec(
        name="read_document", description="d",
        params={"_data": "", "_mime": "", "_name": "", "topk": 3},
    )
    fields = agent.args_schema(spec).model_fields
    assert set(fields) == {"topk"}
    assert agent.args_schema(spec).model_json_schema()["properties"].keys() == {"topk"}


# ---------------------------------------------------------------- 工具兜底

@pytest.mark.anyio
async def test_tool_observation_reports_failure_not_raise():
    async def invoke(tool, image, params):
        raise gw.GatewayError(502, "节点不可达")

    tool = agent.build_tool(ToolSpec(name="x", description="d", needs_image=False), invoke)
    out = await tool.ainvoke({})
    assert out.startswith("工具执行失败")
    assert "GatewayError" in out


# ---------------------------------------------------------------- 结果加工层

def test_render_detect_lists_objects():
    r = {"boxes": [
            {"xyxy": [64.0, 38.4, 332.8, 345.6], "conf": 0.91, "cls": 0, "label": "person"},
            {"xyxy": [371.2, 168.0, 608.0, 326.4], "conf": 0.77, "cls": 2, "label": "car"},
         ], "count": 2, "width": 640, "height": 480}
    text = agent.render_for_llm("detect", r, None)
    assert "共检测到 2 个物体" in text
    assert "person（置信度 0.91）位于 [64, 38, 333, 346]" in text
    assert "car（置信度 0.77）" in text


def test_render_detect_empty():
    text = agent.render_for_llm("detect", {"boxes": [], "count": 0}, None)
    assert "没有检测到" in text


def test_render_ocr_lines():
    r = {"texts": [{"text": "第一行"}, {"text": "第二行"}],
         "full_text": "第一行\n第二行", "count": 2,
         "raw": "第一行\n第二行"}
    text = agent.render_for_llm("ocr", r, None)
    assert "识别出 2 行文字" in text
    assert "1. 第一行" in text and "2. 第二行" in text
    assert "raw" not in text  # 冗余的原始副本不进观察


def test_render_stylize_hides_base64():
    r = {"image": "A" * 5000, "style": "星月夜", "width": 256, "height": 96}
    text = agent.render_for_llm("stylize", r, None)
    assert "星月夜" in text and "256×96" in text and "展示给用户" in text
    assert "AAAA" not in text  # base64 不进观察


def test_render_failure():
    text = agent.render_for_llm("any", None, "RuntimeError: 炸了")
    assert text == "工具执行失败：RuntimeError: 炸了"


def test_render_fallback_strips_raw_and_big_fields():
    """未来新工具没注册渲染器：兜底 JSON 剔 raw、大字段占位。"""
    r = {"raw": "x" * 5000, "items": [{"image": "B" * 5000}], "n": 1}
    text = agent.render_for_llm("future_tool", r, None)
    assert "raw" not in text
    assert "x" * 100 not in text and "B" * 100 not in text
    assert '"n": 1' in text


# ---------------------------------------------------------------- 历史裁剪

def _tool_pair(i: int):
    """一组完整的工具调用对：发起了调用的 AI 消息 + 工具观察。"""
    return [
        AIMessage(content="", tool_calls=[
            {"name": "t", "args": {}, "id": f"c{i}", "type": "tool_call"},
        ]),
        ToolMessage(content="观察", tool_call_id=f"c{i}"),
    ]


def test_trim_history_keeps_pairs_intact():
    """裁剪点落在工具调用对上时，回退到最近的干净边界。"""
    from langchain_core.messages import ToolMessage

    msgs = [HumanMessage(content=f"问{i}") for i in range(4)]
    for i in range(4):
        msgs += _tool_pair(i)
        msgs.append(AIMessage(content=f"答{i}"))
    # 16 条，max=9 → 朴素切割点落在工具对中间
    trimmed = agent.trim_history(msgs, 9)
    # 首条必须是干净边界（不能是工具观察或发起调用的 AI 消息）
    assert trimmed[0].type != "tool"
    assert not getattr(trimmed[0], "tool_calls", None)
    # 不存在「前面没有发起调用的 AI 消息」的孤立 ToolMessage
    for i, m in enumerate(trimmed):
        if m.type == "tool":
            assert trimmed[i - 1].tool_calls, f"孤立 ToolMessage @ {i}"


def test_trim_history_short_passthrough():
    msgs = [HumanMessage(content="a"), AIMessage(content="b")]
    assert agent.trim_history(msgs, 12) == msgs


def test_trim_history_by_char_budget():
    """条数没超但字符超了也要裁——一条带附件正文的消息就能撑爆窗口。

    这是实测踩到的：MAX_HISTORY_MESSAGES 开到 120 也没用，14 条消息里夹两个
    文档块（单条 6000 字）就把 10240 token 的窗口顶穿，vLLM 回 400 打断对话。
    """
    msgs = [
        HumanMessage(content="问一"),
        AIMessage(content="答一"),
        HumanMessage(content="X" * 3000),   # 像带附件正文的那条
    ]
    out = agent.trim_history(msgs, max_messages=99, max_chars=500)
    assert len(out) == 1
    assert out[0].content == "X" * 3000     # 最新的那条必须留下


def test_trim_history_keeps_last_even_if_oversized():
    """单条就超预算时也得留着最后一条：那是用户刚说的话，丢了模型只会答非所问。"""
    msgs = [AIMessage(content="旧的"), HumanMessage(content="X" * 5000)]
    out = agent.trim_history(msgs, max_messages=99, max_chars=100)
    assert len(out) == 1
    assert out[0].content == "X" * 5000


def test_trim_history_char_budget_does_not_override_count_limit():
    """预算宽裕时行为不变：仍按条数上限裁。"""
    msgs = [HumanMessage(content=f"第{i}轮") for i in range(10)]
    out = agent.trim_history(msgs, max_messages=3, max_chars=10**6)
    assert len(out) == 3
    assert out[-1].content == "第9轮"


def test_trim_history_char_budget_counts_multimodal_parts():
    """多模态消息只按 parts 里的文本算字符，不能把图片 base64 也算进去——
    算了的话每条带图消息都会把预算吃光，历史瞬间被清空。"""
    big_image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "z" * 9999}}
    msgs = [
        HumanMessage(content="短"),                       # 1
        HumanMessage(content="中" * 500),                 # 500
        HumanMessage(content=[big_image, {"type": "text", "text": "Y" * 800}]),  # 只算 800
    ]
    # 预算 1300：正确实现能装下后两条（500+800）；若把 base64 也算进去，
    # 光是最后一条就 10799 > 1300，结果会只剩 1 条。
    out = agent.trim_history(msgs, max_messages=99, max_chars=1300)
    assert len(out) == 2
    assert out[-1].content[1]["text"] == "Y" * 800


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


# ---------------------------------------------------------------- 系统提示词

def test_system_prompt_covers_observed_failures():
    """规则里这几条都是照着**实测失败**加的，改提示词时别顺手删掉。"""
    class _Tool:
        def __init__(self, name, description):
            self.name, self.description = name, description

    prompt = agent._system_prompt([
        _Tool("kb_search", "检索百科知识库。用户问百科知识时调用。"),
        _Tool("ocr", "识别图片中的文字。"),
    ])

    # 问「你的知识库到什么时候」时它编了个「2024年12月」——必须明确禁止编日期
    assert "我的知识更新到某年某月" in prompt
    # 追问时它说过「无法调用工具」——必须明确禁止这句
    assert "我无法调用工具" in prompt
    # 禁止编造工具结果（原提示词就有，重构时别丢）
    assert "禁止编造工具结果" in prompt
    # 不交代来历，模型不认「附带文件」那段，还要再调一次 read_document
    assert "【附带文件：" in prompt
    # 工具是动态列的：写死工具名会在节点下线后失真
    assert "kb_search" in prompt and "detect" not in prompt
    # 视觉工具的自主决策策略不能被挤掉
    assert "ocr" in prompt and "stylize" in prompt


def test_no_date_anchor_in_system_prompt():
    """**不要在提示词里写今天的日期**。试过，反而更糟：4B 直接把它当成自己的
    知识截止日吐了出来（"我的知识库更新至2026年10月7日"），比原来那个
    "2024年12月"看起来还权威。"""
    import re

    class _Tool:
        def __init__(self, name, description):
            self.name, self.description = name, description

    prompt = agent._system_prompt([_Tool("kb_search", "检索。")])
    assert not re.search(r"20\d{2}[-年]\d{1,2}", prompt)


def test_kb_triggers_survive_sentence_one_truncation():
    """kb_search 的触发条件必须落在描述的**第一句**里。

    系统提示词列工具时只取 `.split("。")[0]`；触发词原来写在第二句，等于没进
    提示词——实测 4B 就是这么漏掉"关于你自己的问题也要先检索"的。"""
    from kb_node.server import _KB_DESCRIPTION_BASE

    class _Tool:
        def __init__(self, name, description):
            self.name, self.description = name, description

    prompt = agent._system_prompt([_Tool("kb_search", _KB_DESCRIPTION_BASE)])
    assert "关于你自己" in prompt
    assert "时间敏感" in prompt
    assert "核实真假" in prompt
