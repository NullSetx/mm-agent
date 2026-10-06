"""kb_node 测试：切分器 / embedding 客户端（不出网）/ 渲染器 / 入库检索 / 网关发现。

embedding 与 Chroma 全部 fake 或进程内替身，测试不出网；
涉及真实 Chroma 的用例在未安装 chromadb 时自动跳过。
"""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from kb_node import chunker, embed
from kb_node.ingest import ingest
from kb_node.store import KBStore
from llm_node import gateway as gw, llm
from llm_node.agent import render_for_llm
from tests.conftest import make_toy_node

KB = gw.node_url("kb")


# ---------------------------------------------------------------- 切分器

def test_chunk_by_headings():
    md = (
        "# 顶部\n介绍一段。\n\n"
        "## 启动步骤\n先起节点再起网关。\n\n"
        "## 接口\nPOST /api/chat。\n"
    )
    chunks = chunker.chunk_text(md, "docs/a.md")
    sources = [c.source for c in chunks]
    assert "docs/a.md › 顶部" in sources
    assert "docs/a.md › 顶部 › 启动步骤" in sources
    # 前缀进了正文（参与检索），index 全文唯一
    assert all(c.text.startswith(c.source + "\n") for c in chunks)
    assert len({c.index for c in chunks}) == len(chunks)


def test_chunk_no_headings_single_section():
    chunks = chunker.chunk_text("没有标题的一段话。", "b.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "b.txt"


def test_chunk_long_text_splits_with_overlap():
    body = "这是一个很长的句子。" * 60  # 600 字 → 超过单块上限
    chunks = chunker.chunk_text(f"# 长文\n{body}", "c.md")
    assert len(chunks) >= 2
    first = chunks[0].text.split("\n", 1)[1]
    second = chunks[1].text.split("\n", 1)[1]
    assert len(first) <= chunker.MAX_CHARS
    assert second.startswith(first[-chunker.OVERLAP:])  # 相邻块重叠


# ---------------------------------------------------------------- embedding 客户端

def test_embed_api_success_restores_order(monkeypatch):
    monkeypatch.setenv(embed.ENV_API_KEY, "sk-test")
    monkeypatch.setenv(embed.ENV_BASE_URL, "https://api.fake/v1")
    monkeypatch.setenv(embed.ENV_MODEL, "fake-embed")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        # 故意乱序返回，客户端必须按 index 还原
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [3.0]},
            {"index": 0, "embedding": [2.0]},
        ]})

    vectors = embed.embed_texts(
        ["甲", "乙"], client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert vectors == [[2.0], [3.0]]
    assert seen["auth"] == "Bearer sk-test"
    assert seen["url"].endswith("/embeddings")
    assert seen["payload"]["model"] == "fake-embed"


def test_embed_requires_api_key(monkeypatch):
    monkeypatch.delenv(embed.ENV_API_KEY, raising=False)
    with pytest.raises(embed.EmbedError, match="KB_EMBEDDING_API_KEY"):
        embed.embed_texts(["x"])


def test_embed_timeout_maps_to_readable_error(monkeypatch):
    monkeypatch.setenv(embed.ENV_API_KEY, "k")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated", request=request)

    with pytest.raises(embed.EmbedError, match="超时"):
        embed.embed_texts(
            ["x"], client=httpx.Client(transport=httpx.MockTransport(handler))
        )


def test_embed_http_error_maps_to_readable_error(monkeypatch):
    monkeypatch.setenv(embed.ENV_API_KEY, "k")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(embed.EmbedError, match="401"):
        embed.embed_texts(
            ["x"], client=httpx.Client(transport=httpx.MockTransport(handler))
        )


def test_signature_tracks_provider_and_model(monkeypatch):
    monkeypatch.setenv(embed.ENV_PROVIDER, "api")
    monkeypatch.setenv(embed.ENV_MODEL, "BAAI/bge-m3")
    before = embed.signature()
    monkeypatch.setenv(embed.ENV_MODEL, "embedding-3")
    assert embed.signature() != before


# ---------------------------------------------------------------- 观察渲染器

def test_render_kb_hits():
    out = render_for_llm("kb_search", {
        "hits": [{"text": "调一次 POST /api/tools/refresh", "source": "docs/分工与接口约定.md › §5", "score": 0.83}],
        "total": 1,
    })
    assert "知识库" in out
    assert "docs/分工与接口约定.md › §5" in out
    assert "POST /api/tools/refresh" in out


def test_render_kb_miss():
    out = render_for_llm("kb_search", {"hits": [], "total": 0})
    assert "没有检索到" in out


# ---------------------------------------------------------------- mock 分支

def test_kb_search_mock_needs_nothing(monkeypatch):
    """NODE_MOCK=1：不装 chromadb、不配 key 也能返回结构合法的占位结果。"""
    from kb_node import server

    monkeypatch.setenv("NODE_MOCK", "1")
    result = server.kb_search(query="怎么启动")
    assert result["total"] == 1
    assert "[mock]" in result["hits"][0]["text"]
    assert result["hits"][0]["source"]


# ---------------------------------------------------------------- 入库 + 检索（真实 Chroma，未装则跳过）

def _fake_embed_by_keyword(texts, client=None):
    """确定性假向量：文本含「启动」→ [1,0]，否则 [0,1]。"""
    return [[1.0, 0.0] if "启动" in t else [0.0, 1.0] for t in texts]


def test_ingest_and_query_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.setattr(embed, "embed_texts", _fake_embed_by_keyword)

    doc = tmp_path / "guide.md"
    doc.write_text(
        "# 启动\n先起节点再起网关。\n\n# 接口\nPOST /api/chat。\n", encoding="utf-8"
    )
    kb = KBStore(tmp_path / "kb")
    report = ingest([doc], kb)
    assert report["chunks"] == 2
    assert report["total"]["chunks"] == 2

    hits = kb.query([1.0, 0.0], topk=1)  # 「启动」方向的查询
    assert "启动" in hits[0]["source"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)


def test_ingest_incremental(tmp_path, monkeypatch):
    """内容没变的文件跳过不重嵌；改动只重嵌该文件；语料里删掉的文件清块。"""
    pytest.importorskip("chromadb")
    calls: list[list[str]] = []

    def counting_embed(texts, client=None):  # noqa: ANN001, ANN003
        calls.append(list(texts))
        return _fake_embed_by_keyword(texts)

    monkeypatch.setattr(embed, "embed_texts", counting_embed)
    kb = KBStore(tmp_path / "kb")

    guide = tmp_path / "guide.md"
    guide.write_text("# 启动\n先起节点。\n\n# 接口\nPOST /api/chat。\n\n# 其他\n略。\n", encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text("# 启动\n另一篇也讲启动。\n", encoding="utf-8")

    r1 = ingest([guide, other], kb)
    assert r1["chunks"] == 4 and len(r1["changed"]) == 2  # 3 + 1 块，首嵌全量
    assert kb.stats()["chunks"] == 4

    n_calls = len(calls)
    r2 = ingest([guide, other], kb)  # 内容没变 → 一块都不重嵌
    assert r2["chunks"] == 0 and r2["changed"] == [] and len(r2["skipped"]) == 2
    assert len(calls) == n_calls

    # 改 guide 且块数变少（4 → 1）：旧块必须清干净，不能残留
    guide.write_text("# 启动\n改短了。\n", encoding="utf-8")
    r3 = ingest([guide, other], kb)
    assert r3["chunks"] == 1 and len(r3["changed"]) == 1
    assert kb.stats()["chunks"] == 2  # guide 1 块 + other 1 块

    # other 从语料里消失 → 其块清掉
    r4 = ingest([guide], kb)
    assert kb.stats()["chunks"] == 1
    hits = kb.query([1.0, 0.0], topk=10)
    assert all("另一篇" not in h["text"] for h in hits)


def test_ingest_rebuild_overrides_increment(tmp_path, monkeypatch):
    """--rebuild 强制全量重嵌（指纹表作废重建）。"""
    pytest.importorskip("chromadb")
    monkeypatch.setattr(embed, "embed_texts", _fake_embed_by_keyword)
    kb = KBStore(tmp_path / "kb")

    doc = tmp_path / "guide.md"
    doc.write_text("# 启动\n先起节点。\n", encoding="utf-8")
    ingest([doc], kb)
    r = ingest([doc], kb, rebuild=True)
    assert len(r["changed"]) == 1 and r["chunks"] == 1
    assert kb.stats()["chunks"] == 1


def test_query_rejects_embedding_config_change(tmp_path, monkeypatch):
    """入库与查询必须是同一 embedding 配置——换模型后查询直接报可读错误。"""
    pytest.importorskip("chromadb")
    monkeypatch.delenv(embed.ENV_MODEL, raising=False)
    monkeypatch.setattr(embed, "embed_texts", _fake_embed_by_keyword)

    doc = tmp_path / "guide.md"
    doc.write_text("# 启动\n先起节点再起网关。\n", encoding="utf-8")
    kb = KBStore(tmp_path / "kb")
    ingest([doc], kb)

    monkeypatch.setenv(embed.ENV_MODEL, "embedding-3")  # 换了 embedding 模型
    with pytest.raises(RuntimeError, match="重新入库"):
        kb.query([1.0, 0.0], topk=1)


# ---------------------------------------------------------------- 网关发现 + 对话端到端

class FakeAgentModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@pytest.mark.anyio
async def test_gateway_discovers_kb_and_agent_calls_it(gateway, router, monkeypatch):
    """kb 节点上线 → 自动发现；文本提问时 Agent 自主调 kb_search 并照观察回答。"""
    router[KB] = make_toy_node("kb", [
        {"name": "kb_search", "description": "检索项目知识库", "needs_image": False,
         "params": {"query": "", "topk": 3},
         "fn": lambda query="", topk=3: {
             "hits": [{"text": "调一次 POST /api/tools/refresh（或重启网关）",
                       "source": "docs/分工与接口约定.md › §5", "score": 0.9}],
             "total": 1}},
    ])
    report = (await gateway.post("/api/tools/refresh")).json()
    assert report["nodes"]["kb"]["ok"] is True  # kb 节点被发现（vision 节点本测试未挂）
    names = {t["name"] for t in (await gateway.get("/api/tools")).json()["tools"]}
    assert names == {"kb_search"}

    monkeypatch.setattr(
        llm, "build_chat_model",
        lambda **kw: FakeAgentModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "kb_search", "args": {"query": "怎么刷新工具清单"},
                 "id": "k1", "type": "tool_call"},
            ]),
            AIMessage(content="调 POST /api/tools/refresh 即可"),
        ]),
    )
    resp = await gateway.post(
        "/api/chat", json={"session_id": "k1", "message": "怎么刷新工具清单？"}
    )
    body = resp.json()
    assert body["reply"] == "调 POST /api/tools/refresh 即可"
    assert body["tool_calls"][0]["tool"] == "kb_search"
    assert body["tool_calls"][0]["ok"] is True
    assert body["tool_calls"][0]["result"]["total"] == 1
    assert body["tool_calls"][0]["result"]["hits"][0]["source"] == "docs/分工与接口约定.md › §5"
