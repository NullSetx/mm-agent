"""vision_fast 节点：detect + classify。

按文档分工，本节点跑在 8GB #1 上，承载「高频、低延迟」的工具。
两个模型（YOLOv8s / ResNet50）都在**进程启动时加载并常驻显存**，之后不再重载
（见 `models.py`）。

启动：

    uvicorn vision_fast.server:app --host 0.0.0.0 --port 8101

没有模型权重时可以用 mock 模式先把链路跑通（文档 §6.2）：

    NODE_MOCK=1 uvicorn vision_fast.server:app --host 0.0.0.0 --port 8101

绑定 `0.0.0.0` 才能被其他机器的网关访问（文档 §4.1）。
"""

from __future__ import annotations

import logging

import numpy as np

from common.config import mock_enabled
from common.node import build_app
from common.registry import tool
from vision_fast import models

log = logging.getLogger("vision_fast.server")


# ---------------------------------------------------------------- 工具：detect


@tool(
    name="detect",
    description=(
        "目标检测：找出图片里所有物体的位置和类别，返回边界框列表（含每个框的"
        "坐标、置信度和类别名）。适合回答「图里有什么」「在哪里」「有几个」这类问题。"
        "只认 COCO 的 80 类常见物体（人、车、动物、日常用品等），"
        "如果要读图片上的文字请用 ocr，要判断整张图是什么请用 classify。"
    ),
    needs_image=True,
    params={"conf": 0.25, "classes": None},
)
def detect(image: np.ndarray, conf: float = 0.25, classes=None) -> dict:
    if mock_enabled():
        h, w = image.shape[:2]

        def box(fx1, fy1, fx2, fy2, c, cls, label):
            return {
                "xyxy": [round(w * fx1, 1), round(h * fy1, 1),
                         round(w * fx2, 1), round(h * fy2, 1)],
                "conf": c,
                "cls": cls,
                "label": label,
            }

        boxes = [
            box(0.10, 0.08, 0.52, 0.72, 0.91, 0, "person"),
            box(0.58, 0.35, 0.95, 0.68, 0.77, 2, "car"),
        ]
        return {
            "boxes": boxes,
            "count": len(boxes),
            "width": int(w),
            "height": int(h),
            "conf_threshold": float(conf),
            "mock": True,
        }

    return models.detect_objects(image, conf=conf, classes=classes)


# ---------------------------------------------------------------- 工具：classify


@tool(
    name="classify",
    description=(
        "整图分类：判断整张图片属于什么类别，返回 top-k 的候选标签与置信度。"
        "适合回答「这是什么」「什么品种」「什么场景」这类问题。"
        "只给图片级别的一个标签，**不做物体定位**——要找出物体在哪、有几个，"
        "或者要读图上的文字，请分别用 detect 和 ocr。标签为 ImageNet 的 1000 类英文名。"
    ),
    needs_image=True,
    params={"topk": 5},
)
def classify(image: np.ndarray, topk: int = 5) -> dict:
    if mock_enabled():
        predictions = [
            {"index": 285, "label": "Egyptian cat", "score": 0.91},
            {"index": 281, "label": "tabby, tabby cat", "score": 0.06},
            {"index": 282, "label": "tiger cat", "score": 0.02},
        ]
        predictions = predictions[: max(1, min(int(topk), len(predictions)))]
        return {
            "predictions": predictions,
            "top1": predictions[0],
            "count": len(predictions),
            "mock": True,
        }

    return models.classify_image(image, topk=topk)


# ---------------------------------------------------------------- 应用


def _health_detail() -> dict:
    detail = models.pool_status()
    detail["mock"] = mock_enabled()
    return detail


app = build_app("vision-fast", health_hook=_health_detail)

# 文档要求本节点「模型常驻显存」：模块被 uvicorn 导入时就把两个模型加载好，
# 之后所有请求都只做前向计算。mock 模式下跳过，保证没有权重也能秒起。
if not mock_enabled():
    models.load_all()
