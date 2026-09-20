"""vision_heavy 节点：ocr + stylize。

按文档分工，本节点跑在 8GB #2 上，承载「低频、模型偏重」的工具。
两个模型都走懒加载 + 空闲释放（见 `models.py`）。

启动：

    uvicorn vision_heavy.server:app --host 0.0.0.0 --port 8102

没有模型权重时可以用 mock 模式先把链路跑通（文档 §6.2）：

    NODE_MOCK=1 uvicorn vision_heavy.server:app --host 0.0.0.0 --port 8102
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

from common.config import mock_enabled
from common.node import build_app
from common.registry import tool
from vision_heavy import models

log = logging.getLogger("vision_heavy.server")

#: 内置风格图目录。被 .gitignore 的 `data/` 覆盖，各人本地准备
STYLE_DIR = Path(__file__).resolve().parent / "data" / "styles"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
#: 中间图编码质量
JPEG_QUALITY = 92


# ---------------------------------------------------------------- 风格库


def list_styles() -> dict[str, Path]:
    """扫描内置风格库，返回 {风格名: 路径}。风格名即文件名（不含扩展名）。"""
    if not STYLE_DIR.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(STYLE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def _styles_hint() -> str:
    """把可用风格名拼进工具描述，LLM 才知道有哪些合法取值。"""
    names = list(list_styles())
    if not names:
        return "（当前风格库为空，请把风格图放进 vision_heavy/data/styles/ 后重启节点）"
    return "可选风格：" + "、".join(names)


def _encode_jpeg(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("结果图编码为 JPEG 失败")
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- 风格名解析


def _resolve_style(name: str) -> Path:
    styles = list_styles()
    if not styles:
        raise RuntimeError(
            f"风格库为空。请把风格图放进 {STYLE_DIR} 后重启节点"
        )
    if name in styles:
        return styles[name]
    # 容错：LLM 可能带上扩展名，或大小写不一致
    for key, path in styles.items():
        if key.lower() == name.lower() or path.name.lower() == name.lower():
            return path
    raise ValueError(
        f"没有名为 {name!r} 的风格。可用风格：{'、'.join(styles)}"
    )


# ---------------------------------------------------------------- 工具：ocr


@tool(
    name="ocr",
    description=(
        "识别图片中的文字（OCR）。返回识别出的文本内容，"
        "适合读取照片、截图、文档里的文字。支持中英文等多语言。"
    ),
    needs_image=True,
    params={},
)
def ocr(image: np.ndarray) -> dict:
    if mock_enabled():
        return {
            "texts": [{"text": "[mock] 示例文字", "conf": 0.99, "box": [[10, 10], [90, 10], [90, 30], [10, 30]]}],
            "full_text": "[mock] 示例文字",
            "count": 1,
        }

    bundle = models.ocr_model.get()
    return _run_ocr(bundle, image)


def _run_ocr(bundle: dict, image: np.ndarray) -> dict:
    """调用 PaddleOCR-VL 做识别。

    PaddleOCR-VL 是视觉语言模型，输出的是结构化的文本（Markdown/JSON）。
    这里把它的输出归一化成统一的 {texts, full_text, count} 结构，
    这样以后换 OCR 后端（如 PP-OCRv6）时，对网关和 Agent 完全无感。
    """
    import torch
    from PIL import Image

    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # content 里只声明"这里有一张图"，真正的图像数据通过 processor 的 images= 传。
    # 模型的 chat template 就是按这个约定写的（见权重的 chat_template.jinja）。
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "请识别图中的所有文字，按阅读顺序输出。"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[prompt], images=[pil], return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    # 去掉 prompt 部分，只保留新生成的 token
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    return _normalize_ocr_output(raw)


def _normalize_ocr_output(raw: str) -> dict:
    """把 OCR 模型的原始输出整理成统一结构。

    VLM 类 OCR 给的是整段文本（可能带 Markdown），不一定有逐行的置信度和坐标，
    所以 `texts` 里只填能拿到的字段，`full_text` 始终可用。
    """
    text = (raw or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {
        "texts": [{"text": ln} for ln in lines],
        "full_text": text,
        "count": len(lines),
        "raw": text,
    }


# ---------------------------------------------------------------- 工具：stylize


def _stylize_description() -> str:
    return (
        "图像风格迁移：把指定的画风迁移到输入图片上，保持原图内容不变。"
        "输入图片作为内容图，通过 style 参数指定要用的画风。"
        + _styles_hint()
    )


@tool(
    name="stylize",
    description=_stylize_description(),
    needs_image=True,
    params={"style": "", "size": 512, "iters": 100},
)
def stylize(image: np.ndarray, style: str = "", size: int = 512, iters: int = 100) -> dict:
    if mock_enabled():
        h, w = image.shape[:2]
        return {
            "image": _encode_jpeg(image),
            "style": style or "[mock]",
            "width": w,
            "height": h,
            "iters": iters,
            "content_loss": 0.0,
            "style_loss": 0.0,
            "elapsed_s": 0.0,
            "mock": True,
        }

    if not style:
        raise ValueError(
            f"必须通过 style 参数指定画风。{_styles_hint()}"
        )

    style_path = _resolve_style(style)
    style_bgr = cv2.imdecode(np.fromfile(str(style_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if style_bgr is None:
        raise RuntimeError(f"风格图无法解码：{style_path}")

    # 夹一下范围，避免 LLM 传个离谱的值把节点拖死
    size = max(128, min(int(size), 1024))
    iters = max(10, min(int(iters), 400))

    result, info = models.style_transfer(image, style_bgr, size=size, iters=iters)
    return {"image": _encode_jpeg(result), "style": style_path.stem, **info}


# ---------------------------------------------------------------- 应用


def _health_detail() -> dict:
    styles = list(list_styles())
    detail = models.pool_status()
    detail["mock"] = mock_enabled()
    detail["styles"] = styles
    detail["style_dir"] = str(STYLE_DIR)
    return detail


app = build_app("vision-heavy", health_hook=_health_detail)
