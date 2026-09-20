"""vision_heavy 的模型管理：懒加载 + 空闲自动释放显存。

文档给本节点定的策略是「懒加载 + 用完释放显存（低频、模型偏重）」。
本模块提供两层能力：

1. `LazyModel` —— 通用包装器。首次调用才加载，空闲超时后自动卸载并把显存还回去。
2. 两个具体模型的加载逻辑：OCR（PaddleOCR-VL）和风格迁移（VGG19）。

关于「空闲释放」而不是「用完立刻释放」：文档说的是"用完释放"，但如果每次调用后
立刻卸载，连续两次 OCR 就要重新加载两遍模型，延迟无法接受。空闲超时在语义和
实用性之间取平衡，默认 60 秒。

关于可移植性：部署到哪台机器、什么显卡都还没定，所以**不假设任何具体 GPU**。
设备优先 CUDA、失败退 CPU；精度按计算能力在 bf16 / fp16 / fp32 里选，而不是写死。
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("vision_heavy.models")

try:  # torch 是可选的：没装也能起节点跑 mock 模式，方便队友在没有模型的机器上联调
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------- 权重目录

#: 模型权重统一放这里。虽在仓库内，但被 .gitignore 的 `weights/` 覆盖，不会进版本库。
#: 放在仓库内而不是散落在 ~/.cache 里的好处：换机器时整个目录拷过去就能离线启动，
#: 不用再去好几个缓存目录里翻。
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

#: torchvision 的预训练权重缓存位置（VGG19 走这里）。
#: 不改的话它会下到 ~/.cache/torch/hub/checkpoints，和别的项目混在一起。
TORCH_CHECKPOINTS = WEIGHTS_DIR / "checkpoints"

if TORCH_AVAILABLE:
    torch.hub.set_dir(str(WEIGHTS_DIR))


# ---------------------------------------------------------------- 设备与精度


def resolve_device() -> str:
    """选推理设备。优先 CUDA，没有就 CPU。

    可用 `VISION_DEVICE` 强制指定（如 `cpu` / `cuda:0`），部署机显卡情况未知，
    留这个口子方便现场调整。
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
    所以按计算能力逐级退，而不是写死 bf16。
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
    """把缓存显存还给驱动。卸载模型后调用，否则 PyTorch 会攥着不放。"""
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


# ---------------------------------------------------------------- 懒加载包装器

#: 空闲多久后卸掉模型（秒）。可用 VISION_IDLE_TIMEOUT 覆盖；设为 0 表示从不自动卸载
DEFAULT_IDLE_TIMEOUT = float(os.getenv("VISION_IDLE_TIMEOUT", 60))
#: 后台回收线程的检查间隔（秒）
REAP_INTERVAL = 5.0

_POOL: list["LazyModel"] = []
_POOL_LOCK = threading.Lock()
_reaper_started = False


class LazyModel:
    """首次使用时才加载、空闲超时后自动卸载的模型包装器。

    所有公开方法都持锁，避免并发请求把同一个模型加载两遍、或一边用一边被卸载。
    """

    def __init__(
        self,
        name: str,
        loader: Callable[[], Any],
        idle_timeout: float | None = None,
    ) -> None:
        self.name = name
        self._loader = loader
        self._idle_timeout = DEFAULT_IDLE_TIMEOUT if idle_timeout is None else idle_timeout
        self._model: Any = None
        self._last_used = 0.0
        self._load_count = 0
        self._lock = threading.RLock()

    # -------------------------------------------------- 对外

    def get(self) -> Any:
        """取模型；没加载就现在加载。"""
        with self._lock:
            if self._model is None:
                t0 = time.monotonic()
                log.info("加载模型 %s …", self.name)
                self._model = self._loader()
                self._load_count += 1
                log.info("模型 %s 加载完成，耗时 %.1fs", self.name, time.monotonic() - t0)
            self._last_used = time.monotonic()
            return self._model

    def release(self, reason: str = "") -> bool:
        """卸载模型并归还显存。返回是否真的卸了东西。"""
        with self._lock:
            if self._model is None:
                return False
            log.info("卸载模型 %s%s", self.name, f"（{reason}）" if reason else "")
            self._model = None
            gc.collect()
            free_vram()
            return True

    def maybe_release(self) -> bool:
        """空闲超时就卸载。被后台回收线程周期调用。"""
        with self._lock:
            if self._model is None:
                return False
            if self._idle_timeout <= 0:
                return False  # 显式配置为不自动卸载
            if time.monotonic() - self._last_used < self._idle_timeout:
                return False
        return self.release(reason=f"空闲超过 {self._idle_timeout:.0f}s")

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._model is not None
            idle = round(time.monotonic() - self._last_used, 1) if loaded else None
            return {
                "name": self.name,
                "loaded": loaded,
                "load_count": self._load_count,
                "idle_s": idle,
                "idle_timeout_s": self._idle_timeout,
            }


def _register(model: LazyModel) -> LazyModel:
    with _POOL_LOCK:
        _POOL.append(model)
        _start_reaper_locked()
    return model


def _start_reaper_locked() -> None:
    global _reaper_started
    if _reaper_started:
        return
    _reaper_started = True

    def loop() -> None:
        while True:
            time.sleep(REAP_INTERVAL)
            with _POOL_LOCK:
                models = list(_POOL)
            for m in models:
                try:
                    m.maybe_release()
                except Exception as exc:  # noqa: BLE001 - 回收线程不能因为单个失败就死掉
                    log.warning("回收 %s 失败: %s", m.name, exc)

    threading.Thread(target=loop, name="model-reaper", daemon=True).start()


def pool_status() -> dict[str, Any]:
    with _POOL_LOCK:
        models = list(_POOL)
    return {
        "device": resolve_device(),
        "models": [m.status() for m in models],
        "vram": vram_info(),
    }


def release_all() -> None:
    with _POOL_LOCK:
        models = list(_POOL)
    for m in models:
        m.release(reason="主动释放")


# ---------------------------------------------------------------- 风格迁移（Gatys）

# VGG19 的层序号（torchvision features 是顺序结构，靠下标定位）
_CONTENT_LAYERS = [22]                 # relu4_2
_STYLE_LAYERS = [1, 6, 11, 20, 29]     # relu1_1 ~ relu5_1
_MAX_LAYER = max(_CONTENT_LAYERS + _STYLE_LAYERS)
_STYLE_LR = 0.02
_STYLE_INIT_NOISE = 0.02


def _load_vgg19():
    if not TORCH_AVAILABLE:
        raise RuntimeError("未安装 torch，无法进行风格迁移")
    from torchvision.models import VGG19_Weights, vgg19

    device = resolve_device()
    net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    # 截断到 relu5_1：后面的层风格迁移用不到，砍掉省一截计算
    net = net[: _MAX_LAYER + 1].to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


#: 风格迁移用的特征网络。整网只加载一次，被所有迁移请求共享
style_net = _register(LazyModel("vgg19-style", _load_vgg19))


def _to_tensor(bgr: "np.ndarray", max_side: int):
    """numpy BGR → 归一化前的 RGB float 张量 (1,3,H,W)，长边缩到 max_side。"""
    import cv2
    import numpy as np

    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(
            img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
        )
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(resolve_device())


def _to_bgr(t) -> "np.ndarray":
    import cv2
    import numpy as np

    a = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _normalize(x):
    """按 ImageNet 统计量归一化——VGG19 是在这个分布上预训练的，不做这步特征会失真。"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _gram(f):
    """Gram 矩阵：通道两两相关性，捕获纹理/笔触而非内容。"""
    b, c, h, w = f.shape
    f = f.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def _features(net, x, layers):
    out = {}
    for i, layer in enumerate(net):
        x = layer(x)
        if i in layers:
            out[i] = x
    return out


def style_transfer(
    content_bgr: "np.ndarray",
    style_bgr: "np.ndarray",
    size: int = 512,
    iters: int = 100,
    alpha: float = 1.0,
    beta: float = 1e6,
) -> tuple["np.ndarray", dict[str, Any]]:
    """Gatys 风格迁移：固定 VGG19 权重，用梯度下降优化生成图像本身。

    与任务二那版（webapp/tasks/style/engine.py）是同一套算法，区别是输入输出
    走内存里的 numpy 数组——工具收到的是 base64，不落盘。

    返回 (结果 BGR 数组, 统计信息)
    """
    import torch.nn.functional as F

    net = style_net.get()  # 懒加载入口；显存不足会在这里报错

    content = _to_tensor(content_bgr, size)
    style = _to_tensor(style_bgr, size)
    # 风格图与内容图尺寸不必相同，Gram 统计与空间尺寸无关，对齐只是为了流程统一
    if style.shape[-2:] != content.shape[-2:]:
        style = F.interpolate(
            style, size=content.shape[-2:], mode="bilinear", align_corners=False
        )

    with torch.no_grad():
        content_target = _features(net, _normalize(content), _CONTENT_LAYERS)[
            _CONTENT_LAYERS[0]
        ]
        style_feats = _features(net, _normalize(style), _STYLE_LAYERS)
        style_targets = {i: _gram(style_feats[i]) for i in _STYLE_LAYERS}

    # 从内容图加一点噪声起步：比纯噪声收敛快很多，又不会陷入局部解
    gen = (content + _STYLE_INIT_NOISE * torch.randn_like(content)).clamp(0, 1)
    gen.requires_grad_(True)
    opt = torch.optim.Adam([gen], lr=_STYLE_LR)

    t0 = time.monotonic()
    last = {"content": 0.0, "style": 0.0}
    for _step in range(iters):
        feats = _features(net, _normalize(gen), _CONTENT_LAYERS + _STYLE_LAYERS)
        c_loss = F.mse_loss(feats[_CONTENT_LAYERS[0]], content_target)
        s_loss = sum(
            F.mse_loss(_gram(feats[i]), style_targets[i]) for i in _STYLE_LAYERS
        )
        loss = alpha * c_loss + beta * s_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            gen.clamp_(0, 1)

        last = {"content": float(c_loss.detach()), "style": float(s_loss.detach())}

    result = _to_bgr(gen)
    return result, {
        "size": size,
        "iters": iters,
        "alpha": alpha,
        "beta": beta,
        "content_loss": round(last["content"], 5),
        "style_loss": round(last["style"], 9),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "height": int(result.shape[0]),
        "width": int(result.shape[1]),
    }


# ---------------------------------------------------------------- OCR

#: PaddleOCR-VL 的模型 id。部署机可用 OCR_MODEL_ID 换成别的权重。
OCR_MODEL_ID = os.getenv("OCR_MODEL_ID", "PaddlePaddle/PaddleOCR-VL-1.6")
#: 本地权重目录。设了就优先从这里加载，不发任何网络请求。
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH", "")


def default_ocr_dir() -> Path:
    """本地 OCR 权重默认目录，形如 weights/PaddleOCR-VL-1.6"""
    return WEIGHTS_DIR / OCR_MODEL_ID.split("/")[-1]


def resolve_ocr_source() -> str:
    """决定 OCR 权重从哪里加载。

    顺序：`OCR_MODEL_PATH` → 本地缓存目录 → 远程模型 id。

    之所以把「本地目录」放在最前，是因为**部署机不一定能访问外网**：
    把权重预先放好，节点就能完全离线启动，既不需要 HuggingFace 也不需要代理。
    """
    if OCR_MODEL_PATH:
        path = Path(OCR_MODEL_PATH).expanduser()
        if not (path / "config.json").exists():
            raise RuntimeError(f"OCR_MODEL_PATH 指向的目录里没有 config.json：{path}")
        return str(path)

    cached = default_ocr_dir()
    if (cached / "config.json").exists():
        return str(cached)

    # 都没有就只能联网拉了。国内环境建议先手动放到上面那个目录
    return OCR_MODEL_ID


def _load_ocr():
    """加载 PaddleOCR-VL。

    0.9B 的视觉语言模型，fp16 约 2GB，配懒加载在 8GB 节点上很宽松。
    transformers 5.x 原生支持 PaddleOCRVL 架构，不需要 trust_remote_code。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("未安装 torch，无法进行 OCR")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 transformers，无法加载 OCR 模型。"
            "请 pip install transformers，或改用 PP-OCRv6（transformers 同样内置支持）"
        ) from exc

    source = resolve_ocr_source()
    device = resolve_device()
    dtype = resolve_dtype(device)
    log.info("加载 OCR 模型 %s（device=%s, dtype=%s）", source, device, dtype)

    processor = AutoProcessor.from_pretrained(source)
    model = AutoModelForImageTextToText.from_pretrained(
        source, torch_dtype=dtype
    ).to(device).eval()
    return {"model": model, "processor": processor, "device": device, "source": source}


#: OCR 模型。懒加载，空闲后自动释放
ocr_model = _register(LazyModel("paddleocr-vl", _load_ocr))
