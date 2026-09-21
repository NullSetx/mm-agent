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
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from common.schemas import ToolSpec
from llm_node import llm

#: 工具调用循环上限。超出按 GraphRecursionError 处理，给用户可读的回复
MAX_TOOL_ROUNDS = 10

#: 网关注入的调用函数：(tool, image, params) -> InvokeResponse，可抛 GatewayError
InvokeFn = Callable[[str, str | None, dict[str, Any]], Awaitable[Any]]

#: 当前会话图片。请求进来时 set 一次，needs_image 工具执行时读取。
#: 用 ContextVar 而不是工具闭包变量：工具实例跨会话共享，图片是每次请求的
_current_image: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_image", default=None
)

#: 本轮对话的完整工具调用记录（给前端）。工具执行时追加。
#: 与 _current_image 同理：工具实例共享，记录是每次请求的
_current_records: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "current_records", default=None
)

#: 结果里超过这个长度的字符串字段（base64 图片等）不喂给 LLM，
#: 替换成占位符；完整数据仍走 /api/chat 响应给前端
_LLM_FIELD_LIMIT = 4096


def _slim_for_llm(value: Any) -> Any:
    """递归地整理任意工具结果，供"未知工具"的兜底渲染使用：
    - 剔除 `raw` 字段：各工具约定 raw 是后端原始输出副本（如 ocr 的 raw 与
      full_text 逐字重复），喂给 LLM 是纯冗余；
    - 超长字符串（base64 图片等，> _LLM_FIELD_LIMIT）替换为占位符。
    前端拿到的仍是完整数据，裁剪只发生在"给 LLM 的观察文本"这一层。
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "raw":
                continue
            if isinstance(v, str) and len(v) > _LLM_FIELD_LIMIT:
                out[k] = f"<大字段已省略（{len(v)} 字符），已在用户界面展示>"
            else:
                out[k] = _slim_for_llm(v)
        return out
    if isinstance(value, list):
        return [_slim_for_llm(v) for v in value]
    return value


# ---------------------------------------------------------------- 结果加工层

def _render_detect(r: dict[str, Any]) -> str:
    boxes = r.get("boxes") or []
    if not boxes:
        return "检测完成：图中没有检测到目标物体。"
    lines = []
    for b in boxes:
        xyxy = b.get("xyxy") or []
        pos = f"位于 [{', '.join(str(round(c)) for c in xyxy)}]" if len(xyxy) == 4 else ""
        lines.append(f"- {b.get('label', '?')}（置信度 {b.get('conf')}）{pos}")
    size = ""
    if r.get("width") and r.get("height"):
        size = f"图片尺寸 {r['width']}×{r['height']}。"
    return f"共检测到 {len(boxes)} 个物体。{size}\n" + "\n".join(lines)


def _render_ocr(r: dict[str, Any]) -> str:
    lines = [t.get("text", "") for t in (r.get("texts") or []) if isinstance(t, dict)]
    lines = [ln for ln in lines if ln]
    if lines:
        numbered = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
        return f"识别出 {len(lines)} 行文字：\n{numbered}"
    full = (r.get("full_text") or "").strip()
    return f"识别结果：{full}" if full else "识别完成：图中没有文字。"


def _render_classify(r: dict[str, Any]) -> str:
    preds = r.get("predictions") or []
    parts = [
        f"{p.get('label')}（{p.get('score')}）"
        for p in preds[: int(r.get("topk") or len(preds)) or len(preds)]
        if isinstance(p, dict)
    ]
    return "整图分类候选：" + "、".join(parts) if parts else "分类完成，无候选结果。"


def _render_stylize(r: dict[str, Any]) -> str:
    style = r.get("style") or "指定"
    size = ""
    if r.get("width") and r.get("height"):
        size = f"（{r['width']}×{r['height']}）"
    return f"已生成「{style}」风格图片{size}，图片已直接展示给用户。"


#: 已知工具的观察渲染器。未来新增工具若没注册渲染器，走 _render_fallback
_RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "detect": _render_detect,
    "classify": _render_classify,
    "ocr": _render_ocr,
    "stylize": _render_stylize,
}


def render_for_llm(tool_name: str, result: Any, error: str | None = None) -> str:
    """工具结果加工层：把工具返回的 JSON 润色成给 LLM 的中文观察文本。

    小模型读嵌套 JSON 既费 token 又容易编造；润色后的观察文本让它
    "照着念"就能得到正确回答。渲染器缺失或渲染出错时兜底为精简 JSON
    （剔 raw、大字段占位）。
    """
    if error:
        return f"工具执行失败：{error}"
    if result is None:
        return "工具执行完成，没有返回数据。"
    renderer = _RENDERERS.get(tool_name)
    if renderer is not None and isinstance(result, dict):
        try:
            return renderer(result)
        except Exception:  # noqa: BLE001 - 渲染失败不能断对话，兜底精简 JSON
            pass
    return json.dumps(_slim_for_llm(result), ensure_ascii=False)


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
        except Exception as exc:  # noqa: BLE001 - 工具失败要反馈给 LLM 而不是炸掉对话
            err = f"{type(exc).__name__}: {exc}"
            records = _current_records.get()
            if records is not None:
                records.append(
                    {"tool": spec.name, "ok": False, "result": None, "error": err}
                )
            return f"工具执行失败：{err}"

        records = _current_records.get()
        if records is not None:
            records.append(
                {
                    "tool": spec.name,
                    "ok": resp.ok,
                    "result": resp.result,  # 完整数据给前端
                    "error": resp.error,
                }
            )
        # 给 LLM 的是润色后的中文观察（render_for_llm 内部剔 raw、大字段占位）
        return "工具结果：" + render_for_llm(spec.name, resp.result, resp.error)

    return StructuredTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description,
        args_schema=args_schema(spec),
    )


def build_tools(specs: list[ToolSpec], invoke: InvokeFn) -> list[StructuredTool]:
    return [build_tool(spec, invoke) for spec in specs]


# ---------------------------------------------------------------- 对话循环

def _system_prompt(tools: list[StructuredTool]) -> str:
    """按实际可用的工具动态生成系统提示。

    写死工具名会在节点上下线后失真：比如 detect 下线后模型仍被要求
    "看物体用 detect"，调不到就编造检测结果。动态列出真实工具，并明确
    禁止编造，模型缺工具时才会如实说"该功能暂未接入"。
    """
    lines = []
    for t in tools:
        first = (t.description or "").split("。")[0]
        lines.append(f"- {t.name}: {first}。")
    tool_list = "\n".join(lines) if lines else "（当前没有可用工具）"
    return (
        "你是多模态视觉助手，用户可能附带图片，需要你借助视觉工具回答。\n"
        f"当前可用的工具：\n{tool_list}\n\n"
        "规则：\n"
        "1. 图片已经提供给工具，调用工具时不需要、也无法传图片本身；\n"
        "2. 只能调用上面列出的工具，禁止调用列表外的工具，更禁止编造结果；\n"
        "3. 用户消息里以「[xx 观察]」开头的段落，是系统已经替你调用工具得到的"
        "**权威结果**，直接依据它回答，不要怀疑、不要编造观察里没有的信息；\n"
        "4. 用户的问题可能带有错误预设（例如问“有几个人”但观察显示没有人）："
        "一切以观察为准，观察里说没有就明确说没有；\n"
        "5. 图片类工具（如风格迁移）的生成结果已直接展示给用户，你只需一句话"
        "说明生成了什么，不要试图描述图片文件内容；\n"
        "6. 消息里没有对应观察、且列表中没有合适工具时，如实告诉用户该功能"
        "暂未接入，不要假装完成了检测或识别；\n"
        "7. 工具返回里的坐标是像素值，置信度范围 0~1；\n"
        "8. 用中文简洁回答；工具失败时如实告诉用户原因。"
    )


def _human_content(
    message: str, image: str | None, observations: str | None = None
) -> Any:
    """组 HumanMessage 内容。带图时走多模态 parts，纯文本直接用字符串省 token。

    observations 是网关预取的工具观察（见 gateway._prefetch）：
    按成员确认的架构，带图请求由网关先确定性调用分析类工具，
    把润色后的观察文本放进消息，模型依据它回答，不依赖模型自觉调工具。
    """
    if not image and not observations:
        return message
    parts = [{"type": "text", "text": message}]
    if observations:
        parts[0]["text"] += "\n\n" + observations
    if image:
        uri = image if image.startswith("data:") else f"data:image/png;base64,{image}"
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


async def run_chat(
    history: list[BaseMessage],
    message: str,
    image: str | None,
    tools: list[StructuredTool],
    chat_model: Any = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    observations: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[BaseMessage]]:
    """跑一轮对话。

    Args:
        observations: 网关预取的工具观察文本（带图请求由网关先确定性
            调用分析类工具生成），会拼进本轮用户消息。

    Returns:
        (reply, tool_calls, new_messages)：
        reply 是最终回答；tool_calls 是本轮实际发生的工具调用记录
        （文档 §4.3 响应形状）；new_messages 供调用方回填会话历史。
    """
    model = chat_model or llm.build_chat_model()
    agent_app = create_agent(model, tools=tools, system_prompt=_system_prompt(tools))
    human = HumanMessage(
        content=_human_content(message, image, observations)
    )

    token = _current_image.set(image)
    records: list[dict[str, Any]] = []
    token_r = _current_records.set(records)
    try:
        result = await agent_app.ainvoke(
            {"messages": [*history, human]},
            config={"recursion_limit": max_rounds * 2 + 8},
        )
    finally:
        _current_image.reset(token)
        _current_records.reset(token_r)

    all_msgs: list[BaseMessage] = result["messages"]
    new_msgs = all_msgs[len(history) + 1:]  # 截掉输入部分，只留本轮新增

    reply = ""
    for msg in reversed(new_msgs):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # 完整记录（含 base64 图片）由工具执行时经 ContextVar 汇总，供前端渲染；
    # ToolMessage 里的内容是裁剪后的 LLM 视图，不再用于组装响应
    tool_calls = list(records)

    return reply, tool_calls, new_msgs


def trim_history(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """按条数裁剪历史，但绝不把一次工具调用对拦腰切断。

    超限时从尾部保留 max_messages 条；若切割点落在「发起了调用的 AI 消息」
    或「工具观察消息」上，就向前回退到最近的干净边界（用户消息或普通
    AI 回答之后），否则模型会看到一段没有来由的观察文本。
    """
    if len(messages) <= max_messages:
        return list(messages)

    def is_dirty(msg: BaseMessage) -> bool:
        if msg.type == "tool":
            return True
        return bool(getattr(msg, "tool_calls", None))

    start = len(messages) - max_messages
    while start > 0 and is_dirty(messages[start]):
        start -= 1
    return list(messages[start:])


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

