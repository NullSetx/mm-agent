"""接口契约（Pydantic 模型）。

本文件是 `docs/分工与接口约定.md` §4 的代码化表达，字段与文档逐条对应。
**改动本文件需三人同意**（文档 §6.1）。

约定摘要：
- 图片统一走 base64 字符串，允许带 `data:image/...;base64,` 前缀
- 工具内部报错时 HTTP 仍为 200，靠 `ok=false` 表达（不拖垮节点）
- 未知工具由网关/节点返回 404
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """单个工具的自描述。节点靠它对外声明能力，网关靠它构建 LLM 工具。"""

    name: str = Field(description="工具名，全局唯一")
    description: str = Field(description="干什么用，会喂给 LLM 做工具说明")
    needs_image: bool = Field(default=True, description="是否需要输入图片")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="入参名 -> 默认值。默认值同时决定了参数类型，LLM 传错类型时按它强转",
    )


class ToolList(BaseModel):
    """`GET /tools` 的响应。"""

    node: str
    tools: list[ToolSpec] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """`GET /health` 的响应。

    `detail` 留给各节点放自定义信息（如已加载的模型、显存占用），
    不属于契约主体，网关不依赖它的内容。
    """

    node: str
    status: Literal["ok"] = "ok"
    tools: list[str] = Field(default_factory=list, description="已注册的工具名")
    detail: dict[str, Any] = Field(default_factory=dict)


class InvokeRequest(BaseModel):
    """`POST /invoke` 的请求。"""

    tool: str
    image: str | None = Field(default=None, description="base64，可带 data: 前缀；不需要图片的工具可不传")
    params: dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    """`POST /invoke` 的响应。

    成功与失败共用同一结构：失败时 `ok=false`、`result=null`、`error` 为错误描述，
    但 **HTTP 状态码仍是 200**。
    """

    ok: bool
    tool: str
    result: Any = None
    error: str | None = None
    elapsed_ms: float | None = None
