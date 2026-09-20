"""vision_fast 的模型管理：YOLOv8s（检测）+ ResNet50（分类），**进程启动即常驻显存**。

文档给本节点定的策略是「模型常驻显存（高频、要低延迟）」，正好与 vision_heavy 的
「懒加载 + 空闲释放」相反。本模块因此在启动时一次性把两个模型都加载好，之后每次
调用只做前向计算，不再有加载延迟。

为什么必须常驻：本节点承载 detect / classify 这类高频工具，YOLOv8s 每次重新加载
要几百毫秒到数秒，高频场景下完全不可接受；而两个模型都很轻（实测见 README），
在 8GB 卡上留有余量。

可移植性：部署到哪台机器、什么显卡尚未确定，所以**不假设任何具体 GPU**——
设备优先 CUDA、失败退 CPU；精度按计算能力在 bf16 / fp16 / fp32 里逐级退，不写死。
权重一律不进仓库（`.gitignore` 已覆盖 `*.pt` / `*.pth` / `weights/`），路径可用
环境变量覆盖，也支持预先放好让部署机离线启动。

没有装 torch / torchvision / ultralytics 时本模块**仍可导入**：节点能照常起来跑
mock 模式（文档 §6.2），只是两个模型保持未加载，真调用时才报出明确的原因。
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from common.config import WEIGHTS_DIR

log = logging.getLogger("vision_fast.models")

# ---------------------------------------------------------------- 可选依赖

try:  # torch 是可选的：没装也能起节点跑 mock，方便队友在没有模型的机器上联调
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from torchvision.models import ResNet50_Weights, resnet50

    TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    ResNet50_Weights = None  # type: ignore[assignment]
    resnet50 = None  # type: ignore[assignment]
    TORCHVISION_AVAILABLE = False

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]
    ULTRALYTICS_AVAILABLE = False


# ---------------------------------------------------------------- 权重目录

#: 模型权重统一放仓库内 weights/，路径取自 `common.config.WEIGHTS_DIR`（公共配置，
#: 被 .gitignore 覆盖，不会进版本库）。与 vision_heavy/models.py 的约定一致：
#: 换机器时把 weights/ 整个拷过去就能离线启动，不用去 ~/.cache 等好几个缓存目录里翻。

#: torchvision 预训练权重的缓存位置（ResNet50 走这里）。
#: 下面这行把 torchvision 的下载缓存从默认的 ~/.cache/torch/hub/checkpoints
#: 改指到仓库内 weights/checkpoints，保证所有权重集中在一个目录。
TORCH_CHECKPOINTS = WEIGHTS_DIR / "checkpoints"

if TORCH_AVAILABLE:
    torch.hub.set_dir(str(TORCH_CHECKPOINTS))


# ---------------------------------------------------------------- 设备与精度


def resolve_device() -> str:
    """选推理设备。优先 CUDA，没有就 CPU。

    可用 `VISION_DEVICE` 强制指定（如 `cpu` / `cuda:0`）——部署机显卡情况未知，
    留这个口子方便现场调整，行为与 vision_heavy 保持一致。
    """
    forced = os.getenv("VISION_DEVICE", "").strip()
    if forced:
        return forced
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str):
    """按设备能力选精度。

    老显卡可能不支持 bf16（Ampere 之前），更老的甚至不支持 fp16，
    所以按计算能力逐级退，而不是写死。
    """
    if not TORCH_AVAILABLE or not device.startswith("cuda"):
        return torch.float32 if TORCH_AVAILABLE else None
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - 查询失败就按最保守的来
        return torch.float32
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return torch.float32


def free_vram() -> None:
    """把缓存显存还给驱动。仅在主动卸载模型后调用。"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_info() -> dict[str, Any]:
    """当前显存占用，供 /health 的 detail 展示。"""
    if not (TORCH_AVAILABLE and torch.cuda.is_available()):
        return {"available": False}
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "used_mb": round((total_b - free_b) / 2**20, 1),
            "total_mb": round(total_b / 2**20, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- 权重定位


def resolve_yolo_weights() -> str:
    """YOLOv8s 权重路径。

    顺序：`YOLO_WEIGHTS` 环境变量 → `weights/yolov8s.pt` → 交给 ultralytics 下载。

    最后那个兜底会把文件下到**当前工作目录**，`.gitignore` 的 `*.pt` 已覆盖，
    不会污染仓库；但要让部署机离线启动，还是预先放到 `weights/` 或设环境变量。
    """
    env = os.getenv("YOLO_WEIGHTS", "").strip()
    if env:
        return env
    local = WEIGHTS_DIR / "yolov8s.pt"
    if local.exists():
        return str(local)
    return "yolov8s.pt"


def resolve_resnet_weights() -> str:
    """ResNet50 权重来源：本地 `.pth` 路径，或空串表示用 torchvision 预训练权重。

    `CLS_WEIGHTS` 指向本地文件时手动 load_state_dict（部署机离线场景）；
    不设则用 `ResNet50_Weights.IMAGENET1K_V2`，由 torchvision 自动缓存到
    `weights/checkpoints/`（见上方 TORCH_CHECKPOINTS），同样不涉及仓库。
    """
    return os.getenv("CLS_WEIGHTS", "").strip()


# ---------------------------------------------------------------- 常驻模型包装


class ResidentModel:
    """常驻显存的模型包装器。

    与 vision_heavy 的 `LazyModel` 刻意做出区分：**本类不提供空闲卸载**。
    本节点的价值就在于"永不重载"，一旦引入自动卸载，高频调用会被加载延迟拖死。

    `load()` 幂等且**从不抛异常**——启动阶段一个模型加载失败（比如部署机还没放
    权重）不应该让整个节点起不来；错误记在 `status()` 里，`/health` 能直接看到。
    """

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        self.name = name
        self._loader = loader
        self._model: Any = None
        self._error: str | None = None
        self._load_count = 0
        self._load_seconds: float | None = None
        self._lock = threading.RLock()

    def load(self) -> bool:
        """加载模型（幂等）。返回是否处于可用状态。"""
        with self._lock:
            if self._model is not None:
                return True
            t0 = time.monotonic()
            try:
                log.info("加载模型 %s …", self.name)
                self._model = self._loader()
            except Exception as exc:  # noqa: BLE001 - 加载失败不能拖垮节点启动
                self._error = f"{type(exc).__name__}: {exc}"
                log.warning("模型 %s 加载失败：%s", self.name, self._error)
                return False
            self._load_count += 1
            self._load_seconds = round(time.monotonic() - t0, 2)
            self._error = None
            log.info("模型 %s 就绪，耗时 %.2fs", self.name, self._load_seconds)
            return True

    def get(self) -> Any:
        """取模型。若仍未加载会再试一次，失败则抛出带原因的 RuntimeError。"""
        if not self.load():
            raise RuntimeError(
                f"模型 {self.name!r} 不可用（{self._error}）。"
                f"请检查依赖与权重，或置 NODE_MOCK=1 先用 mock 结果联调（文档 §6.2）"
            )
        with self._lock:
            return self._model

    def release(self, reason: str = "") -> bool:
        """主动卸载。正常运行时不需要，留给排查显存问题用。"""
        with self._lock:
            if self._model is None:
                return False
            log.info("卸载模型 %s%s", self.name, f"（{reason}）" if reason else "")
            self._model = None
            gc.collect()
            free_vram()
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "loaded": self._model is not None,
                "load_count": self._load_count,
                "load_seconds": self._load_seconds,
                "error": self._error,
            }


# ---------------------------------------------------------------- 加载实现


def _load_yolo():
    """加载 YOLOv8s。ultralytics 负责下载/读取权重与后处理。"""
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError(
            "未安装 ultralytics，无法加载 YOLOv8s。请 pip install ultralytics"
        )
    weights = resolve_yolo_weights()
    device = resolve_device()
    log.info("加载 YOLOv8s：weights=%s device=%s", weights, device)
    model = YOLO(weights)

    # 把底层 nn.Module 挪到目标设备，让权重在启动时就真正占住显存（这才是"常驻"）。
    # 走 model.model 而不是 model.to()：ultralytics 各版本对 YOLO 对象自身的 .to()
    # 支持不一致，而 .model 始终是标准 nn.Module。推理时仍会传一次 device，
    # 那是给 ultralytics 内部预处理用的，不影响已就位的位置。
    net = getattr(model, "model", None)
    if net is not None and hasattr(net, "to"):
        net.to(device)
    else:  # pragma: no cover - 兼容未来把 model 暴露方式改掉的版本
        log.warning("YOLO 对象没有可用的 .model，跳过显式设备迁移，交给 predict(device=) 处理")

    return {"model": model, "device": device, "weights": weights}


def _load_resnet():
    """加载 ResNet50（ImageNet-1K 预训练）。"""
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "未安装 torchvision，无法加载 ResNet50。请 pip install torchvision"
        )
    device = resolve_device()
    dtype = resolve_dtype(device)
    local = resolve_resnet_weights()

    if local:
        path = Path(local).expanduser()
        if not path.exists():
            raise RuntimeError(f"CLS_WEIGHTS 指向的文件不存在：{path}")
        # 本地权重：结构用官方实现，参数手动灌入
        model = resnet50(weights=None)
        state = torch.load(str(path), map_location="cpu")
        # 兼容有人直接存 state_dict，也有人存 {"state_dict": ...} 的 ckpt
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        source = str(path)
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        source = "torchvision:IMAGENET1K_V2"

    model = model.to(device).eval()
    if dtype is not None and device.startswith("cuda"):
        model = model.to(dtype)

    # 预处理直接用官方 weights 自带的变换：resize/crop/归一化一步到位，
    # 自己手写容易和训练时的分布对不上（本地权重也是同一套 ImageNet 预处理）
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    return {
        "model": model,
        "preprocess": preprocess,
        "categories": list(categories),
        "device": device,
        "dtype": dtype,
        "source": source,
    }


#: 两个常驻模型。模块导入即登记，服务器启动时由 `load_all()` 一次性加载
yolo_model = ResidentModel("yolov8s", _load_yolo)
resnet_model = ResidentModel("resnet50", _load_resnet)
_RESIDENT = [yolo_model, resnet_model]


def load_all() -> dict[str, bool]:
    """把两个常驻模型都加载好。幂等，且从不抛异常。"""
    return {m.name: m.load() for m in _RESIDENT}


def release_all() -> None:
    """主动卸载全部模型。仅排查显存问题时使用。"""
    for m in _RESIDENT:
        m.release(reason="主动释放")


def pool_status() -> dict[str, Any]:
    """模型池状态，喂给 /health 的 detail。"""
    return {
        "device": resolve_device(),
        "models": [m.status() for m in _RESIDENT],
        "vram": vram_info(),
        "yolo_weights": resolve_yolo_weights(),
        "cls_weights": resolve_resnet_weights() or "torchvision:IMAGENET1K_V2",
    }


# ---------------------------------------------------------------- 推理：detect


def detect_objects(
    image: "Any",
    conf: float = 0.25,
    classes: list[int] | None = None,
) -> dict[str, Any]:
    """目标检测。返回 {boxes, count, width, height}。

    `boxes` 每项形如 `{"xyxy": [x1,y1,x2,y2], "conf": 0.87, "cls": 0, "label": "person"}`，
    坐标是原图像素坐标（ultralytics 已按原图比例还原）。
    """
    bundle = yolo_model.get()
    model = bundle["model"]
    device = bundle["device"]

    # 夹一下范围，避免 LLM 传个离谱的值把节点拖死
    conf = max(0.0, min(float(conf), 1.0))
    class_filter = None
    if classes:
        # LLM 可能给字符串或单个数字，统一成 int 列表
        class_filter = [int(c) for c in classes] if not isinstance(classes, (int, str)) else [int(classes)]

    t0 = time.monotonic()
    results = model.predict(
        source=image,
        conf=conf,
        classes=class_filter,
        device=device,
        verbose=False,
    )
    elapsed_s = round(time.monotonic() - t0, 3)

    boxes: list[dict[str, Any]] = []
    height, width = int(image.shape[0]), int(image.shape[1])
    if results:
        r = results[0]
        names = getattr(r, "names", {}) or {}
        xyxy = getattr(r.boxes, "xyxy", None)
        if xyxy is not None and len(xyxy):
            coords = xyxy.tolist()
            confs = r.boxes.conf.tolist()
            clss = r.boxes.cls.tolist()
            for (x1, y1, x2, y2), c, k in zip(coords, confs, clss):
                idx = int(k)
                boxes.append(
                    {
                        "xyxy": [round(float(x1), 1), round(float(y1), 1),
                                 round(float(x2), 1), round(float(y2), 1)],
                        "conf": round(float(c), 4),
                        "cls": idx,
                        "label": str(names.get(idx, idx)),
                    }
                )

    return {
        "boxes": boxes,
        "count": len(boxes),
        "width": width,
        "height": height,
        "conf_threshold": conf,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------- 推理：classify


def classify_image(image: "Any", topk: int = 5) -> dict[str, Any]:
    """整图分类。返回 {predictions, top1, count}。

    `predictions` 每项形如 `{"index": 285, "label": "Egyptian cat", "score": 0.91}`，
    标签来自 ImageNet-1K 的英文类别名。
    """
    import cv2
    from PIL import Image

    bundle = resnet_model.get()
    model = bundle["model"]
    device = bundle["device"]
    dtype = bundle["dtype"]
    categories = bundle["categories"]

    topk = max(1, min(int(topk), 20))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = bundle["preprocess"](Image.fromarray(rgb)).unsqueeze(0).to(device)
    if dtype is not None and device.startswith("cuda"):
        tensor = tensor.to(dtype)

    t0 = time.monotonic()
    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(dim=-1)[0]
        top = probs.topk(min(topk, probs.numel()))
    elapsed_s = round(time.monotonic() - t0, 3)

    predictions = [
        {
            "index": int(i),
            "label": categories[int(i)] if int(i) < len(categories) else str(int(i)),
            "score": round(float(p), 4),
        }
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    return {
        "predictions": predictions,
        "top1": predictions[0] if predictions else None,
        "count": len(predictions),
        "elapsed_s": elapsed_s,
    }
