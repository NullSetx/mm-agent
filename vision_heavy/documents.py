"""文档读取：PDF 逐页取文本层，扫描页回退 OCR；文本 / 代码文件直接解码。

**为什么先取文本层而不是无脑 OCR**：电子版 PDF 自带精确的字符流，直接读出来又快
又准（一页几毫秒 vs 走一遍 OCR 十几秒，而且 OCR 会把标点和生僻字认错）。所以顺序是
「先试文本层，只有这页确实没有文本层（扫描件）才渲染成图交给 OCR」。

PyMuPDF 是可选依赖：没装时 PDF 走不了，但节点照常启动、`read_document` 返回可读
的报错（与 kb_node 对 chromadb 的处理一致）。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import cv2
import numpy as np

log = logging.getLogger("vision_heavy.documents")

#: 文本层少于这么多字符时**才需要进一步判断**是不是扫描页（见 _looks_scanned）。
#: 不能拿它当"少于这个数就是扫描件"——只有标题的一页也会很短，那种页送去 OCR
#: 纯属白等十几秒。
TEXT_LAYER_MIN_CHARS = 20

#: 一张图占整页面积超过这个比例、且页面文字很少，就判定为扫描页
SCANNED_IMAGE_RATIO = 0.6

#: 一次最多处理多少页：防止一份几百页的 PDF 把节点占住，也让耗时可控。
#: 有文本层的页几乎不花时间，所以这个上限只对"页数"生效。
MAX_PAGES = 20

#: **扫描页**（要过 OCR）的上限，单独卡得更紧：OCR 一页约 16 秒，首次还要
#: 先加载 1.8GB 的模型，而工具调用的超时只有 30 秒（TOOL_TIMEOUT）。
#: 演示时想多读几页扫描件，就把 TOOL_TIMEOUT 和这个值一起调大。
MAX_SCANNED_PAGES = int(os.getenv("DOC_MAX_SCANNED_PAGES", "2"))

#: 返回文本的字符上限。网关会把它拼进对话上下文并**随历史留存**，而模型的窗口
#: 只有 VLLM_MAX_MODEL_LEN（默认 4096，本机 10240）。定太大就等于让一份附件把
#: 整个上下文吃光，前面聊过的全被挤掉，所以这里给得比较克制。
#: 需要读全份长文档时调大它，并同时把网关的 MAX_HISTORY_CHARS 一起调大。
MAX_CHARS = 3000

#: 扫描页渲染成图时的 DPI。150 够 OCR 认字，再高只是白费时间。
RENDER_DPI = 150

#: 按内容解码文本文件时依次尝试的编码。gb18030 覆盖 GBK/GB2312，
#: Windows 上导出的 .txt/.csv 经常是这套。
TEXT_ENCODINGS = ("utf-8", "gb18030")

PDF_MAGIC = b"%PDF-"


def pymupdf_available() -> bool:
    """PDF 支持是否可用。PyMuPDF 是可选依赖，没装只影响 PDF，不挡节点启动。"""
    try:
        import pymupdf  # noqa: F401
        return True
    except ImportError:
        return False


def sniff(data: bytes) -> str:
    """按**内容**判断类型，返回 'pdf' / 'image' / 'text' / 'binary'。

    不看扩展名也不信 mime：上传路径上这两样都可能是错的（或干脆没有），
    而 PDF 和常见图片都有固定的文件头，判定比它们可靠。
    """
    if data[:5] == PDF_MAGIC:
        return "pdf"
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None:
        return "image"
    # NUL 字节：文本文件不会有，有就是二进制。这条必须挡在解码之前——
    # gb18030 的字节覆盖极广，几乎任何二进制都能"解出"一堆乱码，
    # 那些乱码喂给模型只会带偏它。
    if b"\x00" in data[:4096]:
        return "binary"
    if _decode_text(data) is not None:
        return "text"
    return "binary"


def _decode_text(data: bytes) -> str | None:
    """尽力解码成文本；都不成返回 None。"""
    for enc in TEXT_ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _page_to_bgr(page: Any) -> np.ndarray:
    """把 PDF 的一页渲染成 OpenCV 约定的 BGR 数组（OCR 的入参格式）。"""
    pix = page.get_pixmap(dpi=RENDER_DPI)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _clip(text: str) -> tuple[str, bool]:
    """按字符数截断，尽量切在换行处而不是句子中间。"""
    if len(text) <= MAX_CHARS:
        return text, False
    cut = text[:MAX_CHARS].rfind("\n")
    return text[: cut if cut > MAX_CHARS * 0.6 else MAX_CHARS], True


def _looks_scanned(page: Any) -> bool:
    """这一页是不是扫描件（没有可用文本层，得靠 OCR）。

    只看"文字非空"会把带水印 / 页眉的扫描件误判成文本页，结果正文一个字都读不到；
    只看"字数够多"又会让只有标题的幻灯片被白送去 OCR。所以两步走：
    文字够多 → 肯定是文本页；文字很少 → 再看是不是有张图几乎盖满整页。
    """
    text = (page.get_text("text") or "").strip()
    if not text:
        return True
    if len(text) >= TEXT_LAYER_MIN_CHARS:
        return False

    rect = page.rect
    page_area = abs(rect.width * rect.height) or 1.0
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if abs((x1 - x0) * (y1 - y0)) / page_area > SCANNED_IMAGE_RATIO:
            return True
    return False


def read_pdf(data: bytes, ocr_fn: Callable[[np.ndarray], dict[str, Any]] | None) -> dict[str, Any]:
    """读 PDF。`ocr_fn` 是节点的 OCR 回调，只对没有文本层的页调用。

    返回 {text, pages, pages_total, scanned_pages, truncated, note}。
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "读 PDF 需要 pymupdf，请先安装：pip install -r requirements.txt"
        ) from exc

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        total = doc.page_count
        pages = min(total, MAX_PAGES)
        chunks: list[str] = []
        scanned = 0
        skipped_scanned = 0

        for i in range(pages):
            page = doc[i]
            text = (page.get_text("text") or "").strip()
            if _looks_scanned(page):
                # 没有可用的文本层，渲染成图交给 OCR
                if ocr_fn is None:
                    raise RuntimeError("这一页没有文本层（扫描件），需要 OCR 才能读")
                if scanned >= MAX_SCANNED_PAGES:
                    skipped_scanned += 1
                    continue
                scanned += 1
                result = ocr_fn(_page_to_bgr(page)) or {}
                text = str(result.get("full_text") or "").strip()
            if text:
                chunks.append(f"--- 第 {i + 1} 页 ---\n{text}")

        body, truncated = _clip("\n\n".join(chunks))
        notes = []
        if total > pages:
            notes.append(f"共 {total} 页，只读了前 {pages} 页")
        if scanned:
            notes.append(f"其中 {scanned} 页是扫描件，走的是 OCR")
        if skipped_scanned:
            notes.append(
                f"另有 {skipped_scanned} 页是扫描件，超过单次上限（{MAX_SCANNED_PAGES} 页）没有读"
            )
        if truncated:
            notes.append(f"内容较长，已截到 {MAX_CHARS} 字")
        return {
            "text": body,
            "pages": pages,
            "pages_total": total,
            "scanned_pages": scanned,
            "truncated": truncated,
            "note": "；".join(notes),
        }
    finally:
        doc.close()


def read_document(
    data: bytes,
    ocr_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """统一入口：按内容分派到 PDF / 图片 / 文本。

    返回 {kind, text, ..., note}——`text` 是给模型看的正文。
    """
    kind = sniff(data)

    if kind == "pdf":
        out = read_pdf(data, ocr_fn)
        out["kind"] = "pdf"
        return out

    if kind == "image":
        if ocr_fn is None:
            raise RuntimeError("这是一张图片，需要 OCR 才能读，但节点的 OCR 不可用")
        text = str((ocr_fn(_decode_image(data)) or {}).get("full_text") or "").strip()
        body, truncated = _clip(text)
        return {
            "kind": "image", "text": body, "pages": 1, "pages_total": 1,
            "scanned_pages": 1, "truncated": truncated,
            "note": "图片走的是 OCR" + ("；内容较长已截断" if truncated else ""),
        }

    if kind == "text":
        raw = _decode_text(data) or ""
        body, truncated = _clip(raw)
        return {
            "kind": "text", "text": body, "pages": 1, "pages_total": 1,
            "scanned_pages": 0, "truncated": truncated,
            "note": "内容较长，已截断" if truncated else "",
        }

    raise ValueError("不认识的文件类型：既不是 PDF / 图片，也不是能解码的文本")


def _decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("不是有效图片")
    return img
