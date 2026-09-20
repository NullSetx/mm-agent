"""节点外壳：把工具注册表包装成符合契约的 HTTP 服务。

两个视觉节点（vision-fast / vision-heavy）用完全相同的这套外壳，
实现差异全在各自治的 `server.py` 里。网关按 `/tools` 自动发现工具，
所以**加新工具不需要改网关和 Agent 的任何代码**（文档 §5）。

暴露的端点（文档 §4.2）：
    GET  /health   存活 + 已注册工具名
    GET  /tools    自报工具清单
    POST /invoke   统一调用入口
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from common.registry import ToolRegistry, ToolError, registry as default_registry
from common.schemas import HealthResponse, InvokeRequest, InvokeResponse, ToolList

log = logging.getLogger("common.node")


def decode_image(data: str) -> np.ndarray:
    """base64 → numpy BGR 数组（cv2 约定）。

    按文档 §4.1，图片允许带 `data:image/...;base64,` 前缀；实际传过来的字符串
    还常见缺 padding、带换行等情况，这里一并处理。
    """
    if not data or not data.strip():
        raise ValueError("image 字段为空")

    text = data.strip()
    if text.startswith("data:"):
        # data:image/jpeg;base64,<payload>
        _, sep, payload = text.partition(",")
        if not sep:
            raise ValueError("data URI 缺少逗号分隔符")
        text = payload

    text = "".join(text.split())  # 去掉换行/空格
    # base64 长度必须是 4 的倍数，不足补 '='
    padding = -len(text) % 4
    if padding:
        text += "=" * padding

    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image 不是合法的 base64：{exc}") from exc

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("base64 解出的内容不是有效图片")
    return img


def build_app(
    node_name: str,
    registry: ToolRegistry | None = None,
    health_hook: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    """构建一个节点的 FastAPI 应用。

    Args:
        node_name: 节点名，会出现在 /health 和 /tools 的响应里
        registry: 工具注册表。默认用全局注册表（各节点 server.py 往它上面挂工具）
        health_hook: 返回额外健康信息的回调，内容会放进 HealthResponse.detail
    """
    reg = registry if registry is not None else default_registry
    if not reg.node:
        reg.node = node_name

    app = FastAPI(title=f"{node_name} node", version="1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        detail: dict[str, Any] = {}
        if health_hook is not None:
            try:
                detail = health_hook() or {}
            except Exception as exc:  # noqa: BLE001 - 健康检查本身不该把节点搞挂
                log.warning("health_hook 执行失败: %s", exc)
                detail = {"health_hook_error": f"{type(exc).__name__}: {exc}"}
        return HealthResponse(node=node_name, status="ok", tools=reg.names(), detail=detail)

    @app.get("/tools", response_model=ToolList)
    def tools() -> ToolList:
        return ToolList(node=node_name, tools=reg.specs())

    @app.post("/invoke", response_model=InvokeResponse)
    def invoke(req: InvokeRequest) -> Any:
        started = time.perf_counter()

        found = reg.get(req.tool)
        if found is None:
            # 未知工具是路由问题，不是工具执行问题，按文档 §4.3 返回 404
            raise HTTPException(
                status_code=404,
                detail=f"节点 {node_name} 上没有名为 {req.tool!r} 的工具，"
                       f"可用工具：{reg.names()}",
            )

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        image = None
        if found.spec.needs_image:
            if req.image is None:
                return InvokeResponse(
                    ok=False, tool=req.tool, result=None,
                    error="该工具需要图片，但请求里没有 image 字段",
                    elapsed_ms=elapsed(),
                )
            try:
                image = decode_image(req.image)
            except ValueError as exc:
                return InvokeResponse(
                    ok=False, tool=req.tool, result=None,
                    error=f"ValueError: {exc}", elapsed_ms=elapsed(),
                )

        try:
            result = found.call(image, req.params or {})
        except Exception as exc:  # noqa: BLE001 - 工具内部错误不能拖垮节点
            log.exception("工具 %s 执行失败", req.tool)
            return InvokeResponse(
                ok=False, tool=req.tool, result=None,
                error=f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed(),
            )

        log.info("工具 %s 执行成功，耗时 %.1fms", req.tool, elapsed())
        return InvokeResponse(
            ok=True, tool=req.tool, result=result, error=None, elapsed_ms=elapsed()
        )

    @app.exception_handler(ToolError)
    async def _tool_error(_request, exc: ToolError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
