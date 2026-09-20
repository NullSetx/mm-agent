"""LangChain Agent 编排（文档分工表 A 区：统一网关、Agent 编排）。

从聚合工具清单（common.schemas.ToolSpec）动态构建 LangChain 工具，
用 langchain 1.x 的 create_agent 跑「LLM → tool_calls → 执行 → 回填 → 再问」
的循环，最多 MAX_TOOL_ROUNDS 轮，防 LLM 死循环。

与图片相关的关键设计：**图片不进 LLM 的参数**。needs_image 工具的
args schema 里没有 image 字段，LLM 只决定「调什么工具、传什么普通参数」；
真正执行时由网关把会话当前图片（ContextVar 里的）注入 InvokeRequest。
这样既避免 LLM 复读几十 KB 的 base64，也避免它把图片弄丢。

mock 模式（NODE_MOCK=1，文档 §6.2「先跑通 mock 再上模型」）走 mock_chat，
不依赖任何 LLM。
"""

from __future__ import annotations

import contextvars
import json
from typing import Any, Awaitable, Callable

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from common.schemas import ToolSpec
from llm_node import llm

#: 工具调用循环上限。超出按 GraphRecursionError 处理，给用户可读的回复
MAX_TOOL_ROUNDS = 5

#: 网关注入的调用函数：(tool, image, params) -> InvokeResponse，可抛 GatewayError
InvokeFn = Callable[[str, str | None, dict[str, Any]], Awaitable[Any]]

#: 当前会话图片。请求进来时 set 一次，needs_image 工具执行时读取。
#: 用 ContextVar 而不是工具闭包变量：工具实例跨会话共享，图片是每次请求的
_current_image: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_image", default=None
)


# ---------------------------------------------------------------- 工具构建

_SIMPLE_TYPES: dict[type, type] = {bool: bool, int: int, float: float, str: str}


def args_schema(spec: ToolSpec) -> type[BaseModel]:
    """把 ToolSpec.params（参数名 -> 默认值）转成 Pydantic args schema。

    默认值即类型声明（common/registry.py 的同一约定）：默认值为 None 的参数
    （如 detect 的 classes）无法推断类型，按可选 Any 处理。
    无参数的工具也要给显式空对象 schema——若留空，LangChain 会从 **kwargs
    签名推断出一个 "kwargs" 属性喂给 LLM，把它带偏。
    """
    model = create_model(f"{spec.name}_args", **{
        name: (
            _SIMPLE_TYPES.get(type(default), Any),
            Field(default=default, description=f"默认 {default!r}"),
        )
        for name, default in spec.params.items()
    })
    return model


def build_tool(spec: ToolSpec, invoke: InvokeFn) -> StructuredTool:
    """把一个 ToolSpec 包成 LangChain 工具。

    工具对 LLM 的输出是 JSON 字符串（含 ok/error），LLM 据此决定继续调工具
    还是给出最终回答。执行异常在这里兜底转成 ok=false，不中断整轮对话。
    """

    async def _run(**kwargs: Any) -> str:
        image = _current_image.get() if spec.needs_image else None
        try:
            resp = await invoke(spec.name, image, kwargs)
            payload = {
                "tool": spec.name,
                "ok": resp.ok,
                "result": resp.result,
                "error": resp.error,
            }
        except Exception as exc:  # noqa: BLE001 - 工具失败要反馈给 LLM 而不是炸掉对话
            payload = {
                "tool": spec.name,
                "ok": False,
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return json.dumps(payload, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description,
        args_schema=args_schema(spec),
    )


def build_tools(specs: list[ToolSpec], invoke: InvokeFn) -> list[StructuredTool]:
    return [build_tool(spec, invoke) for spec in specs]


# ---------------------------------------------------------------- 对话循环

SYSTEM_PROMPT = (
    "你是多模态视觉助手，用户可能附带图片，需要你借助视觉工具回答。\n"
    "规则：\n"
    "1. 图片已经提供给工具，调用工具时不需要、也无法传图片本身；\n"
    "2. **凡涉及图片内容的问题（有什么物体、位置、数量、文字、类别等），"
    "必须先调用对应工具拿到结果，再根据结果回答；禁止不调工具直接描述图片**："
    "看物体用 detect，判断整图类别用 classify，读文字用 ocr，改画风用 stylize；\n"
    "3. 工具返回里的坐标是像素值，置信度范围 0~1；\n"
    "4. 用中文简洁回答；工具失败时如实告诉用户原因。"
)


def _human_content(message: str, image: str | None) -> Any:
    """组 HumanMessage 内容。带图时走多模态 parts，纯文本直接用字符串省 token。"""
    if not image:
        return message
    uri = image if image.startswith("data:") else f"data:image/png;base64,{image}"
    return [
        {"type": "text", "text": message},
        {"type": "image_url", "image_url": {"url": uri}},
    ]


async def run_chat(
    history: list[BaseMessage],
    message: str,
    image: str | None,
    tools: list[StructuredTool],
    chat_model: Any = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, list[dict[str, Any]], list[BaseMessage]]:
    """跑一轮对话。

    Returns:
        (reply, tool_calls, new_messages)：
        reply 是最终回答；tool_calls 是本轮实际发生的工具调用记录
        （文档 §4.3 响应形状）；new_messages 供调用方回填会话历史。
    """
    model = chat_model or llm.build_chat_model()
    agent = create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)
    human = HumanMessage(content=_human_content(message, image))

    token = _current_image.set(image)
    try:
        result = await agent.ainvoke(
            {"messages": [*history, human]},
            config={"recursion_limit": max_rounds * 2 + 8},
        )
    finally:
        _current_image.reset(token)

    all_msgs: list[BaseMessage] = result["messages"]
    new_msgs = all_msgs[len(history) + 1:]  # 截掉输入部分，只留本轮新增

    reply = ""
    for msg in reversed(new_msgs):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    tool_calls: list[dict[str, Any]] = []
    for msg in new_msgs:
        if not isinstance(msg, ToolMessage):
            continue
        record: dict[str, Any] = {"tool": msg.name}
        try:
            payload = json.loads(msg.content)
            record["ok"] = bool(payload.get("ok"))
            record["result"] = payload.get("result")
            if payload.get("error"):
                record["error"] = payload["error"]
        except (TypeError, ValueError):
            record.update(ok=False, result=str(msg.content))
        tool_calls.append(record)

    return reply, tool_calls, new_msgs


# ---------------------------------------------------------------- mock 模式

async def mock_chat(
    message: str, image: str | None, specs: list[ToolSpec]
) -> tuple[str, list[dict[str, Any]]]:
    """NODE_MOCK=1 时的假对话：不依赖 LLM，返回结构合法的占位结果。"""
    names = [s.name for s in specs]
    tool_calls: list[dict[str, Any]] = []
    if image and names:
        first = next((s for s in specs if s.needs_image), specs[0])
        tool_calls.append({"tool": first.name, "ok": True, "result": {"mock": True}})
    reply = f"[mock] 已收到消息（可用工具：{'、'.join(names) if names else '无'}）"
    if tool_calls:
        reply += f"，并模拟调用了 {tool_calls[0]['tool']}"
    return reply, tool_calls
