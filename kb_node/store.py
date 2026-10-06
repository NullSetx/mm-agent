"""Chroma 向量库封装：入库重建 / 相似检索，持久化到 data/kb/（方案 §3.3）。

chromadb 是 kb_node 的重依赖，刻意懒加载：未安装时节点照常起、/health
照常绿（detail 里给提示），mock 模式完全不碰它。cosine 距离换算成相似度
score = 1 - distance 返回。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common.config import DATA_DIR
from kb_node.chunker import Chunk
from kb_node import embed

#: 集合名。一套语料（百科词条 + 项目文档），不需要多集合
COLLECTION = "project_docs"

#: 入库指纹表文件名（存在库目录里）：{文件路径: {hash, chunks}}，增量入库用
MANIFEST = "manifest.json"


def _chunk_id(c: Chunk) -> str:
    """稳定 id：同一文档同一位置重入库时 upsert 覆盖而非追加。"""
    return hashlib.sha1(f"{c.source}#{c.index}".encode("utf-8")).hexdigest()


class KBStore:
    """一个 Chroma 持久化目录的封装。path 可注入（测试用 tmp 目录）。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DATA_DIR / "kb"
        self._client: Any = None

    # ---------------------------------------------------------- 基础

    def _chroma(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "知识库需要 chromadb：pip install chromadb"
                    "（requirements.txt kb_node 分区，A 的机器装）"
                ) from exc
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _new_collection(self) -> Any:
        """按当前 embedding 配置建集合，签名写进 metadata 供查询时校验。"""
        return self._chroma().get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine", "embedding": embed.signature()},
        )

    # ---------------------------------------------------------- 入库

    def rebuild(self) -> Any:
        """删掉旧集合并按当前配置新建空集合。全量重建 / 配置变更时用。"""
        client = self._chroma()
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001 - 首次入库时集合还不存在
            pass
        return self._new_collection()

    def load_manifest(self) -> dict[str, dict[str, Any]]:
        """文件内容指纹表。损坏 / 缺失（首次入库、手动删库）按空表处理。"""
        try:
            data = json.loads((self.path / MANIFEST).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 空表是常态而非错误
            return {}
        return data if isinstance(data, dict) else {}

    def save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def open_for_write(self, force_rebuild: bool = False) -> tuple[Any, dict[str, dict[str, Any]]]:
        """拿一个可写的集合与指纹表。三种情况清空重来：强制重建 / 集合不存在 /
        embedding 配置变了（换模型的向量空间不通用，manifest 一并作废）；
        否则返回现有集合与指纹表，调用方走增量。"""
        client = self._chroma()
        manifest = self.load_manifest()
        if not force_rebuild:
            try:
                col = client.get_collection(COLLECTION)
                sig = (col.metadata or {}).get("embedding")
                if sig and sig == embed.signature():
                    return col, manifest
            except Exception:  # noqa: BLE001 - 还没建过集合 → 走重建
                pass
        col = self.rebuild()
        self.save_manifest({})
        return col, {}

    @staticmethod
    def delete_file(col: Any, name: str) -> None:
        """清掉一个文件的全部旧块。增量入库改文件前先清，防止块数变少后残留。"""
        col.delete(where={"file": name})

    @staticmethod
    def upsert(col: Any, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        col.upsert(
            ids=[_chunk_id(c) for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "file": c.file} for c in chunks],
        )
        return len(chunks)

    # ---------------------------------------------------------- 查询

    def collection(self) -> Any:
        """取现有集合做查询。空库 / embedding 配置与入库时不一致都直接报错。"""
        try:
            col = self._chroma().get_collection(COLLECTION)
        except Exception:  # noqa: BLE001 - 不存在/没装 chromadb 统一转可读错误
            raise RuntimeError("知识库为空，请先入库：python -m kb_node.ingest") from None
        sig = (col.metadata or {}).get("embedding")
        if sig and sig != embed.signature():
            raise RuntimeError(
                f"知识库由 embedding「{sig}」构建，当前配置是「{embed.signature()}」，"
                "两者向量空间不通用。请切回原配置，或重新入库：python -m kb_node.ingest"
            )
        return col

    def query(self, vector: list[float], topk: int = 3) -> list[dict[str, Any]]:
        """cosine 相似检索，返回 [{text, source, score}]，score=1-distance。"""
        col = self.collection()
        total = col.count()
        if total == 0:
            return []
        res = col.query(
            query_embeddings=[vector],
            n_results=max(1, min(int(topk), total)),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": doc,
                "source": (meta or {}).get("source", ""),
                "score": round(1.0 - float(dist), 4),
            }
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def stats(self) -> dict[str, Any]:
        """/health 与工具描述用的概览；不抛错（未入库/未装 chromadb 给空态）。"""
        try:
            col = self._chroma().get_collection(COLLECTION)
            chunks = col.count()
            sig = (col.metadata or {}).get("embedding")
        except Exception:  # noqa: BLE001 - 空态是常态而非错误
            return {"chunks": 0, "embedding": embed.signature()}
        return {"chunks": chunks, "embedding": sig or embed.signature()}


_default: KBStore | None = None


def default_store() -> KBStore:
    """进程级共享的默认库（data/kb/）。"""
    global _default
    if _default is None:
        _default = KBStore()
    return _default
