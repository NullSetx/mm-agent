"""入库 CLI：把语料（百科词条 + 项目文档）向量化写进 Chroma（方案 §4 左半边）。

用法：

    python -m kb_node.ingest                # 默认收 wiki/、docs/ 全部 + README.md
    python -m kb_node.ingest wiki/          # 显式指定文件或目录
    python -m kb_node.ingest --rebuild      # 不走增量，全部重嵌

默认**增量**：按内容指纹（sha256）跳过没变的文件，只重嵌改动过的；语料里
删掉的文件其旧块也会一并清掉。embedding 配置变更（换模型/供应商）或结果
可疑时加 --rebuild 全量重建。中断后重跑即可，最坏情况用 --rebuild 兜底。
api 模式需要 KB_EMBEDDING_API_KEY；入库与查询必须是同一 embedding 配置。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from common.config import ROOT
from kb_node import chunker, embed, store

#: 可入库的文本格式
TEXT_EXTS = {".md", ".txt"}


def collect_files(paths: list[Path]) -> list[Path]:
    """展开目录（递归），过滤出文本文件，去重保序。"""
    files: dict[Path, None] = {}
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in TEXT_EXTS:
                    files.setdefault(f, None)
        elif p.is_file() and p.suffix.lower() in TEXT_EXTS:
            files.setdefault(p, None)
    return list(files)


def _corpus_name(f: Path) -> str:
    """文件在语料里的名字：相对仓库根优先（显示成 wiki/xx.md 这类短路径）。"""
    try:
        return f.relative_to(ROOT).as_posix()
    except ValueError:
        return f.as_posix()


def ingest(
    paths: list[Path], kb: store.KBStore | None = None, rebuild: bool = False
) -> dict[str, Any]:
    """切分 → 向量化 → 入库（默认增量）。

    增量规则：文件内容指纹没变 → 跳过（不调 embedding）；变了 → 清旧块重嵌；
    上次入库有、这次语料里没有的文件 → 清块。返回报告
    （files/changed/skipped/chunks/dim/total）。
    """
    kb = kb or store.default_store()
    files = collect_files(paths)
    col, manifest = kb.open_for_write(force_rebuild=rebuild)

    names = [_corpus_name(f) for f in files]
    corpus = dict(zip(names, files))

    # 上次收过、这次语料里没有的文件：块清掉，指纹作废
    for gone in [n for n in manifest if n not in corpus]:
        kb.delete_file(col, gone)
        del manifest[gone]

    changed: list[str] = []
    skipped: list[str] = []
    done, dim = 0, 0
    for f, name in zip(files, names):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if manifest.get(name, {}).get("hash") == digest:
            skipped.append(name)
            continue

        chunks = chunker.chunk_file(f, name)
        batches: list[tuple[list[chunker.Chunk], list[list[float]]]] = []
        for i in range(0, len(chunks), embed.BATCH):
            batch = chunks[i:i + embed.BATCH]
            vectors = embed.embed_texts([c.text for c in batch])
            dim = len(vectors[0])
            batches.append((batch, vectors))

        # 先嵌完再动库：embedding 挂了不丢旧数据。入库中途崩了指纹不落盘，
        # 重跑会重新嵌这个文件（浪费但正确），实在不行 --rebuild 全量兜底
        kb.delete_file(col, name)
        for batch, vectors in batches:
            done += kb.upsert(col, batch, vectors)

        changed.append(name)
        if chunks:
            manifest[name] = {"hash": digest, "chunks": len(chunks)}
        else:
            manifest.pop(name, None)

    kb.save_manifest(manifest)
    return {
        "files": names, "changed": changed, "skipped": skipped,
        "chunks": done, "dim": dim, "total": kb.stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="百科知识库入库（默认增量）")
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="文件或目录（.md/.txt），默认 wiki/、docs/ 与 README.md",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="不走增量，全部重嵌（换过 embedding 模型 / 结果可疑时用）",
    )
    args = parser.parse_args()
    paths = args.paths or [
        p for p in (ROOT / "wiki", ROOT / "docs", ROOT / "README.md") if p.exists()
    ]

    try:
        report = ingest(paths, rebuild=args.rebuild)
    except embed.EmbedError as exc:
        print(f"[入库失败] {exc}", file=sys.stderr)
        return 1
    if not report["chunks"] and not report["skipped"]:
        print(f"[入库失败] 没有切出任何内容：{report['files'] or paths}", file=sys.stderr)
        return 1
    total = report["total"]
    print(
        f"[入库完成] 新嵌 {len(report['changed'])} 篇 → {report['chunks']} 块，"
        f"跳过未变 {len(report['skipped'])} 篇"
        f"（维度 {report['dim']}；库内共 {total['chunks']} 块，embedding {total['embedding']}）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
