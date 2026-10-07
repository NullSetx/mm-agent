"""kb_node 服务：`kb_search` 工具 + 节点外壳（方案 §5）。

与视觉节点同构：`@tool` 登记进注册表，`build_app` 提供 /health /tools
/invoke。kb_search 是纯文本工具（needs_image=False），网关自动发现后
Agent 在对话里自主调用，检索百科词条与项目文档回答知识类问题。

启动：uvicorn kb_node.server:app --host 0.0.0.0 --port 8103
入库：python -m kb_node.ingest            # 默认收 wiki/、docs/ 与 README.md
"""

from __future__ import annotations

from typing import Any

from common.config import mock_enabled
from common.node import build_app
from common.registry import tool

# 注意：**触发条件必须写进第一句**。系统提示词列工具时只取描述的第一句
# （agent._system_prompt 的 `.split("。")[0]`），写在第二句等于没进提示词——
# 实测 4B 就是这样漏掉"关于你自己的问题也要先检索"这条的。
_KB_DESCRIPTION_BASE = (
    "检索百科知识库，**凡是需要事实的问题都先调它**——百科概念（某概念是什么、"
    "原理 / 用法 / 对比）、核实真假（是不是、真的吗）、时间敏感（最新 / 最近 / "
    "现在 / 有哪些）、本项目自身（怎么启动、谁负责什么、接口参数），"
    "以及**关于你自己**的问题（你能做什么、你的知识更新到什么时候）。"
    "收录 wiki/ 百科词条 + docs/ 项目文档。看图分析、闲聊不要调用。"
)


def _kb_description() -> str:
    """动态描述（stylize 同款技巧）：带上已收录规模，帮模型判断覆盖面。

    chromadb 未装或还没入库时不能阻塞节点启动（mock-first），静默退回
    基础描述。
    """
    try:
        from kb_node import store

        n = store.default_store().stats()["chunks"]
    except Exception:  # noqa: BLE001 - 描述只是锦上添花，失败不挡启动
        return _KB_DESCRIPTION_BASE
    if n:
        return f"{_KB_DESCRIPTION_BASE}当前收录 {n} 个文档片段。"
    return _KB_DESCRIPTION_BASE + "（知识库暂为空，请先运行 python -m kb_node.ingest）"


@tool(
    name="kb_search",
    description=_kb_description(),
    needs_image=False,
    params={"query": "", "topk": 3},
)
def kb_search(query: str, topk: int = 3) -> dict[str, Any]:
    """检索百科知识库，返回最相关的词条/文档片段（含来源）。"""
    if mock_enabled():
        return {
            "query": query,
            "hits": [{
                "text": "[mock] 入库后这里是最相关的词条正文。",
                "source": "wiki/mock.md › 示例词条",
                "score": 0.9,
            }],
            "total": 1,
        }
    if not query or not query.strip():
        raise ValueError("query 不能为空：请把要查的问题写成检索词")

    from kb_node import embed, store

    topk = max(1, min(int(topk), 10))
    vector = embed.embed_texts([query.strip()])[0]
    hits = store.default_store().query(vector, topk)
    return {"query": query, "hits": hits, "total": len(hits)}


def _health_detail() -> dict[str, Any]:
    """/health 的 detail：收录规模 + 当前 embedding 配置。"""
    try:
        from kb_node import store

        return store.default_store().stats()
    except Exception as exc:  # noqa: BLE001 - 健康检查不能把节点搞挂
        return {"error": f"{type(exc).__name__}: {exc}"}


app = build_app("kb", health_hook=_health_detail)
