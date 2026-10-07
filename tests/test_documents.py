"""documents 测试：类型嗅探、PDF 文本层优先、扫描页回退 OCR、截断与各种上限。

PDF 用 PyMuPDF 现造。pymupdf 是可选依赖，没装就整个文件跳过。
不出网、不加载任何模型——OCR 用假回调，只记录被调了几次、入参是什么。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_heavy import documents

pymupdf = pytest.importorskip("pymupdf")


def _fake_ocr(text: str = "OCR 出来的文字"):
    """假 OCR：记录每次调用的入参形状。"""
    calls: list[tuple] = []

    def fn(img: np.ndarray) -> dict:
        calls.append(img.shape)
        return {"full_text": text}

    return fn, calls


def _pdf(pages: list[tuple[str, bool]]) -> bytes:
    """造 PDF。pages 每项 = (页面上的文本, 是否再贴一张图)。

    "有图无文本" 就等于扫描页——这正是要区分开的两种情况。
    """
    doc = pymupdf.open()
    for text, with_image in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontsize=16)
        if with_image:
            img = np.full((300, 800, 3), 255, np.uint8)
            cv2.putText(img, "SCANNED", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 4)
            ok, buf = cv2.imencode(".png", img)
            assert ok
            page.insert_image(pymupdf.Rect(40, 40, 460, 200), stream=buf.tobytes())
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_fullpage_image(header: str) -> bytes:
    """整页铺一张大图 + 页眉一小行字：典型的水印 / 扫描件。"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), header, fontsize=11)
    img = np.full((600, 400, 3), 255, np.uint8)
    cv2.putText(img, "BODY", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    page.insert_image(page.rect, stream=buf.tobytes())
    data = doc.tobytes()
    doc.close()
    return data


def _png() -> bytes:
    img = np.full((80, 200, 3), 255, np.uint8)
    cv2.putText(img, "HELLO", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------- 类型嗅探

def test_sniff_by_content_not_extension():
    assert documents.sniff(_pdf([("Hi", False)])) == "pdf"
    assert documents.sniff(_png()) == "image"
    assert documents.sniff("def f():\n    return 1\n".encode()) == "text"


def test_sniff_rejects_binary():
    """含 NUL 的内容一律当二进制——gb18030 能把几乎任何字节解成乱码，
    不拦住的话模型会收到一坨乱码正文。"""
    assert documents.sniff(b"\x00\x01\x02\xff\xfe\xfd") == "binary"
    assert documents.sniff(b"\x7fELF\x02\x01\x01" + b"\x00" * 32) == "binary"


# ---------------------------------------------------------------- PDF

def test_digital_pdf_uses_text_layer_and_skips_ocr():
    """电子版 PDF 直接读文本层，OCR 一次都不该被调用——这是整件事的主要收益：
    又快又准，还不占显存。"""
    ocr, calls = _fake_ocr()
    out = documents.read_pdf(_pdf([("Hello mm-agent", False)]), ocr)
    assert calls == []
    assert "Hello mm-agent" in out["text"]
    assert out["scanned_pages"] == 0


def test_scanned_page_falls_back_to_ocr():
    """没有文本层的页才渲染成图交给 OCR，且必须给 BGR 三通道（OCR 的入参约定）。"""
    ocr, calls = _fake_ocr("扫描页正文")
    out = documents.read_pdf(_pdf([("", True)]), ocr)
    assert len(calls) == 1
    assert calls[0][2] == 3
    assert "扫描页正文" in out["text"]
    assert out["scanned_pages"] == 1
    assert "OCR" in out["note"]


def test_short_title_page_does_not_trigger_ocr():
    """只有一行标题的页不该被送去 OCR——那是十几秒的浪费。
    这是"文字少"不能直接等于"扫描件"的原因。"""
    ocr, calls = _fake_ocr()
    out = documents.read_pdf(_pdf([("Chapter 1", False)]), ocr)
    assert calls == []
    assert "Chapter 1" in out["text"]


def test_watermarked_scan_is_treated_as_scanned():
    """页眉一行字 + 整页图片：只看"文字非空"会漏掉正文，必须判为扫描页。"""
    ocr, calls = _fake_ocr("正文来自 OCR")
    out = documents.read_pdf(_pdf_fullpage_image("Confidential"), ocr)
    assert len(calls) == 1
    assert "正文来自 OCR" in out["text"]
    assert out["scanned_pages"] == 1


def test_mixed_pdf_only_ocrs_the_scanned_page():
    ocr, calls = _fake_ocr("扫描件")
    out = documents.read_pdf(_pdf([("Text page", False), ("", True)]), ocr)
    assert len(calls) == 1              # 只对第二页走了 OCR
    assert "Text page" in out["text"]
    assert "扫描件" in out["text"]


def test_scanned_pages_are_capped():
    """扫描页要过 OCR（一页十几秒），必须单独卡上限，否则必然撞工具超时。"""
    ocr, calls = _fake_ocr("x")
    out = documents.read_pdf(_pdf([("", True)] * 4), ocr, )
    assert len(calls) == documents.MAX_SCANNED_PAGES
    assert "超过单次上限" in out["note"]


def test_page_cap_and_note():
    out = documents.read_pdf(_pdf([("A", False)] * 25), None)
    assert out["pages"] == documents.MAX_PAGES
    assert out["pages_total"] == 25
    assert "只读了前" in out["note"]


def test_scanned_page_without_ocr_raises():
    with pytest.raises(RuntimeError, match="OCR"):
        documents.read_pdf(_pdf([("", True)]), None)


# ---------------------------------------------------------------- 文本 / 其它

def test_text_file_decodes_and_gb18030_fallback():
    out = documents.read_document("def f():\n    return 1\n".encode())
    assert out["kind"] == "text"
    assert "def f()" in out["text"]

    out = documents.read_document("中文内容，用 GBK 存".encode("gb18030"))
    assert "中文内容" in out["text"]


def test_long_text_is_truncated(monkeypatch):
    monkeypatch.setattr(documents, "MAX_CHARS", 200)
    out = documents.read_document(("这一行有点长。\n" * 200).encode())
    assert out["truncated"] is True
    assert len(out["text"]) <= 200
    assert "截断" in out["note"]


def test_image_goes_through_ocr():
    ocr, calls = _fake_ocr("图上的字")
    out = documents.read_document(_png(), ocr)
    assert out["kind"] == "image"
    assert len(calls) == 1
    assert "图上的字" in out["text"]


def test_unknown_binary_rejected():
    with pytest.raises(ValueError, match="不认识"):
        documents.read_document(b"\x00\x01\xff\xfe")


def test_pymupdf_available_reports_bool():
    assert documents.pymupdf_available() is True
