"""中文文档切分器：markdown 标题定章节、句子边界定块（方案 §3.4）。

每块的 text 带「文档名 › 章节路径」前缀入库——前缀本身参与检索，能明显
提高「问启动步骤」这类问题的命中率；source 单独存元数据，供回答标注出处。
纯文本（无 markdown 标题）整体视为一个章节。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: 单块正文上限（字）。方案 §3.4：300~500
MAX_CHARS = 500
#: 相邻块重叠字数
OVERLAP = 50

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENT_END = "。！？；\n"


@dataclass(frozen=True)
class Chunk:
    """一个入库块。source + index 构成稳定 id，重入库时 upsert 覆盖。"""

    text: str    # 入库正文：`文档名 › 章节路径\n内容`
    source: str  # `docs/xx.md › §4.3 网关对外接口`
    index: int   # 文件内序号
    file: str = ""  # 所属文件的语料内路径，增量入库按它整文件清旧块


def chunk_text(text: str, doc_name: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """把一篇文档切成 Chunk 列表。"""
    chunks: list[Chunk] = []
    for path, body in _sections(text):
        if not body.strip():
            continue
        source = f"{doc_name} › {path}" if path else doc_name
        for part in _pack(_sentences(body), max_chars, OVERLAP):
            chunks.append(
                Chunk(text=f"{source}\n{part}", source=source,
                      index=len(chunks), file=doc_name)
            )
    return chunks


def chunk_file(path: Path | str, doc_name: str | None = None) -> list[Chunk]:
    """读文件并切分。doc_name 默认用 posix 相对/绝对路径（入库元数据的来源名）。"""
    p = Path(path)
    name = doc_name or p.as_posix()
    return chunk_text(p.read_text(encoding="utf-8", errors="replace"), name)


# ---------------------------------------------------------------- 内部实现

def _sections(text: str) -> list[tuple[str, str]]:
    """按 markdown 标题把文档切成 (章节路径, 正文)。"""
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []   # 未闭合的标题层级 [(级别, 标题)]
    body: list[str] = []

    def close() -> None:
        path = " › ".join(title for _, title in stack)
        out.append((path, "\n".join(body)))
        body.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            close()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    close()
    return out


def _sentences(text: str) -> list[str]:
    """按句末标点（。！？；）和换行切分，标点跟随句尾；空白段丢弃。"""
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENT_END:
            parts.append(text[start:i + 1])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


def _pack(parts: list[str], max_chars: int, overlap: int) -> list[str]:
    """把句子装进 ≤max_chars 的块，相邻块带 ~overlap 字重叠。"""
    # 超过单块上限的长句（无标点的代码块、表格等）先硬切
    flat: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            flat.append(p)
            continue
        step = max(1, max_chars - overlap)
        for i in range(0, len(p), step):
            flat.append(p[i:i + max_chars])
            if i + max_chars >= len(p):
                break

    chunks: list[str] = []
    cur = ""
    for p in flat:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            seed = cur[-overlap:] if overlap else ""
            cur = seed + p if len(seed) + len(p) <= max_chars else p
        else:
            cur += p
    if cur.strip():
        chunks.append(cur)
    return chunks
