"""工具注册表。

每个工具只声明四件事（文档 §5）：`name` / `description` / `needs_image` / `params`。
节点用 `@tool` 装饰器登记，节点外壳再从注册表生成 `/tools` 响应和 `/invoke` 分发。

用法：

    @tool(name="ocr", description="识别图片中的文字", needs_image=True,
          params={"lang": "ch"})
    def ocr(image, lang="ch"):
        return {"texts": [...]}

两个和 LLM 协作相关的处理（直接决定 Agent 能不能正确调用工具）：

1. **参数过滤**：LLM 有时会臆造出契约里没声明的参数，只透传声明过的，避免
   `TypeError: unexpected keyword argument`。
2. **类型强转**：LLM 经常把数字发成字符串（`{"conf": "0.5"}`），按 `params` 里
   默认值的类型转换。注意 `bool` 是 `int` 的子类，判断顺序必须 bool 在前。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from common.schemas import ToolSpec

#: 带图工具的函数签名里，图片形参必须叫这个名字
IMAGE_PARAM = "image"


class ToolError(RuntimeError):
    """工具登记或调用过程中的契约性错误。"""


def _coerce(value: Any, default: Any, name: str) -> Any:
    """把 LLM 传来的值转成默认值所声明的类型。

    默认值为 None 时无法推断类型，原样透传（文档 §5 里 `classes: null` 这种）。
    转换失败抛 ToolError 而不是静默用默认值——静默会把 LLM 的调用错误藏起来，
    报出来 Agent 才有机会纠正。
    """
    if default is None:
        return value
    try:
        # bool 必须最先判断：Python 里 bool 是 int 的子类，顺序反了 True/False 会被当成 1/0
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(default, int):
            if isinstance(value, str):
                text = value.strip()
                try:
                    return int(text)
                except ValueError:
                    return int(float(text))  # 兼容 "3.0" 这种
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return value if isinstance(value, str) else str(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(
            f"参数 {name!r} 的值 {value!r} 无法转换为 {type(default).__name__}"
        ) from exc
    # list / dict 等复合类型不做转换，原样透传
    return value


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    fn: Callable[..., Any]

    def call(self, image: Any, params: dict[str, Any]) -> Any:
        """按契约调用工具：过滤参数 → 类型强转 → 调用。"""
        declared = self.spec.params
        kwargs = {
            key: _coerce(value, declared[key], key)
            for key, value in params.items()
            if key in declared  # 不在契约里的参数直接丢弃
        }
        if self.spec.needs_image:
            return self.fn(**{IMAGE_PARAM: image}, **kwargs)
        return self.fn(**kwargs)


@dataclass
class ToolRegistry:
    """一个节点持有的工具集合。

    独立成类而不是纯模块级字典，是为了让测试能建互不干扰的隔离实例。
    """

    node: str = ""
    _tools: dict[str, RegisteredTool] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        needs_image: bool = True,
        params: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        params = dict(params or {})

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ToolError(f"工具名重复：{name!r}（工具名必须全局唯一）")

            # 登记时就校验签名，把拼错参数名这类问题拦在启动阶段，
            # 而不是等 LLM 调用时才发现参数被静默丢弃
            sig = inspect.signature(fn)
            if needs_image and IMAGE_PARAM not in sig.parameters:
                raise ToolError(
                    f"工具 {name!r} 声明 needs_image=True，"
                    f"但函数签名里没有名为 {IMAGE_PARAM!r} 的形参"
                )
            unknown = set(params) - set(sig.parameters)
            if unknown:
                raise ToolError(
                    f"工具 {name!r} 声明的参数 {sorted(unknown)} 不在函数签名里，"
                    f"签名只有 {sorted(sig.parameters)}"
                )

            self._tools[name] = RegisteredTool(
                spec=ToolSpec(
                    name=name,
                    description=description,
                    needs_image=needs_image,
                    params=params,
                ),
                fn=fn,
            )
            return fn

        return decorate

    # ---------------------------------------------------------- 查询

    def all(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def clear(self) -> None:
        self._tools.clear()


#: 默认注册表。各节点的 server.py 直接往它上面挂工具。
registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    needs_image: bool = True,
    params: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """把一个函数登记成工具（挂到默认注册表）。"""
    return registry.register(name, description, needs_image, params)
