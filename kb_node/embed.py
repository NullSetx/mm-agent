"""Embedding 客户端：OpenAI 兼容 /embeddings API 为主，本地模型兜底（方案 §3.2）。

- 默认 SiliconFlow `BAAI/bge-m3`（免费）；智谱 embedding-3 等同协议服务改
  环境变量即可切换。API key 只走环境变量 / .env，绝不进仓库；
- **入库与查询必须同一 provider+model**：`signature()` 标识当前配置，
  store 层据此拦截「向量空间不一致」的查询；
- 失败抛 `EmbedError` → 节点包装成 ok=false（HTTP 仍 200），对话不崩；
- api 模式只用 httpx（公共依赖，零新增）；local 模式才懒加载
  sentence-transformers（可选依赖，未装时报可读错误）。
"""

from __future__ import annotations

import os
from typing import Any

import httpx

ENV_PROVIDER = "KB_EMBEDDING_PROVIDER"    # api（默认）| local
ENV_BASE_URL = "KB_EMBEDDING_BASE_URL"    # 默认 SiliconFlow
ENV_API_KEY = "KB_EMBEDDING_API_KEY"
ENV_MODEL = "KB_EMBEDDING_MODEL"          # 默认 BAAI/bge-m3
ENV_TIMEOUT = "KB_EMBEDDING_TIMEOUT"      # 秒
ENV_LOCAL_MODEL = "KB_LOCAL_EMBEDDING_MODEL"  # 默认 BAAI/bge-small-zh-v1.5

_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
#: 单次请求批量（各 embedding API 普遍限 16~32 条）
BATCH = 16


class EmbedError(RuntimeError):
    """embedding 调用失败（缺 key / 网络 / 上游错误）。"""


def provider() -> str:
    return os.getenv(ENV_PROVIDER, "api").strip().lower() or "api"


def model_id() -> str:
    default = _DEFAULT_LOCAL_MODEL if provider() == "local" else _DEFAULT_MODEL
    return os.getenv(ENV_MODEL if provider() == "api" else ENV_LOCAL_MODEL, default)


def signature() -> str:
    """当前 embedding 配置的标识。入库时写进库，查询时不一致即拒绝。"""
    return f"{provider()}:{model_id()}"


def embed_texts(texts: list[str], client: httpx.Client | None = None) -> list[list[float]]:
    """把一批文本向量化。client 仅供测试注入（httpx.MockTransport），生产自建。"""
    if not texts:
        return []
    if provider() == "local":
        return _embed_local(texts)
    return _embed_api(texts, client)


# ---------------------------------------------------------------- api 模式

def _embed_api(texts: list[str], client: httpx.Client | None) -> list[list[float]]:
    api_key = os.getenv(ENV_API_KEY, "").strip()
    if not api_key:
        raise EmbedError(
            f"缺少 {ENV_API_KEY}（写进 .env 或环境变量）。"
            "没有 key 可置 NODE_MOCK=1 先跑通全链路，或 KB_EMBEDDING_PROVIDER=local 用本地模型"
        )
    base = os.getenv(ENV_BASE_URL, _DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv(ENV_TIMEOUT, "10"))

    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            vectors.extend(_post_embeddings(client, base, api_key, batch))
        return vectors
    finally:
        if own:
            client.close()


def _post_embeddings(
    client: httpx.Client, base: str, api_key: str, batch: list[str]
) -> list[list[float]]:
    try:
        resp = client.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model_id(), "input": batch},
        )
    except httpx.TimeoutException as exc:
        raise EmbedError(f"embedding API 超时（{ENV_TIMEOUT} 默认 10s）：{base}") from exc
    except httpx.HTTPError as exc:
        raise EmbedError(f"embedding API 不可达：{type(exc).__name__}: {exc}") from exc

    if resp.status_code != 200:
        raise EmbedError(f"embedding API HTTP {resp.status_code}：{resp.text[:200]}")
    try:
        data: list[dict[str, Any]] = resp.json()["data"]
        if len(data) != len(batch):
            raise EmbedError(f"embedding API 返回 {len(data)} 条，请求 {len(batch)} 条")
        return [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]
    except EmbedError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbedError(f"embedding API 响应格式异常：{exc}") from exc


# ---------------------------------------------------------------- local 模式

_LOCAL_MODEL: Any = None


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _LOCAL_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbedError(
            "KB_EMBEDDING_PROVIDER=local 需要 sentence-transformers："
            "pip install sentence-transformers（requirements.txt kb 分区，可选）"
        ) from exc
    if _LOCAL_MODEL is None:
        _LOCAL_MODEL = SentenceTransformer(model_id(), device="cpu")
    return _LOCAL_MODEL.encode(texts, normalize_embeddings=True).tolist()
