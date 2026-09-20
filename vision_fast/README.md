# vision_fast 节点

`detect` + `classify` 两个工具。按文档分工跑在 **8GB #1** 节点上。

本节点面向**高频、低延迟**场景，所以两个模型都**在进程启动时加载并常驻显存**，
之后每次调用只做前向计算，不再有加载延迟（与 `vision_heavy` 的
「懒加载 + 空闲释放」刻意相反）。

| 工具 | 模型 | 用途 |
|---|---|---|
| `detect` | YOLOv8s | 目标检测，返回边界框列表（坐标 / 置信度 / 类别名） |
| `classify` | ResNet50 | 整图分类，返回 top-k 标签与置信度 |

## 启动

```bash
# 从仓库根目录
pip install -r requirements.txt

# 本节点的额外依赖默认是注释掉的，需要自行安装（见 requirements.txt 的 B 分区）
pip install torch torchvision ultralytics

# 正常启动
uvicorn vision_fast.server:app --host 0.0.0.0 --port 8101

# 没有模型权重时，用 mock 模式先把链路跑通（文档 §6.2）
NODE_MOCK=1 uvicorn vision_fast.server:app --host 0.0.0.0 --port 8101
```

绑定 `0.0.0.0` 才能被其他机器的网关访问（文档 §4.1）。

## 自检

```bash
python -m common.selftest     # 契约自检，不需要模型权重，18 项
curl localhost:8101/health    # 看设备、显存、两个模型的加载状态
curl localhost:8101/tools     # 看工具清单
```

调一次 `detect`（不用手搓 base64）：

```bash
python - <<'PY'
import base64, json, cv2, numpy as np, httpx

img = np.zeros((480, 640, 3), np.uint8)
img[80:340, 60:330] = (180, 160, 150)
ok, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf.tobytes()).decode()

r = httpx.post("http://127.0.0.1:8101/invoke",
               json={"tool": "detect", "image": b64, "params": {"conf": 0.25}},
               timeout=30)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
PY
```

## 权重准备

权重不进仓库（`.gitignore` 已覆盖 `*.pt` / `*.pth` / `weights/`）。两个模型的获取方式不同：

### detect 用的 YOLOv8s

按以下顺序查找：

1. 环境变量 `YOLO_WEIGHTS` 指向的文件
2. `weights/yolov8s.pt`
3. 都没有则交给 `ultralytics` 自动下载（**会落到当前工作目录**，`*.pt` 已被 gitignore，
   不污染仓库）

推荐把权重预先放到 `weights/yolov8s.pt`，部署机即可完全离线启动：

```bash
mkdir -p weights && cd weights
curl -sSL -O https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
```

### classify 用的 ResNet50

默认用 `torchvision` 的 `ResNet50_Weights.IMAGENET1K_V2`，**首次加载时自动下载**到
`~/.cache/torch/hub/checkpoints/`（约 98MB），无需手动操作。

要离线部署，就把 `.pth` 放到任意位置并用 `CLS_WEIGHTS` 指过去：

```bash
# 在能联网的机器上先导出
python -c "import torch; from torchvision.models import resnet50, ResNet50_Weights; \
torch.save(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict(), 'resnet50.pth')"

# 部署机
export CLS_WEIGHTS=/path/to/resnet50.pth
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `NODE_MOCK` | 空 | 置 `1` 时工具返回占位结果，用于无权重联调 |
| `VISION_DEVICE` | 自动 | 强制设备，如 `cpu` / `cuda:0`。不设则优先 CUDA、失败退 CPU |
| `YOLO_WEIGHTS` | 空 | YOLOv8s 权重路径，优先级最高 |
| `CLS_WEIGHTS` | 空 | ResNet50 权重（`.pth`）路径；不设则用 torchvision 预训练权重 |
| `VISION_FAST_HOST` | `127.0.0.1` | 供网关发现本节点（`common/config.py` 读取） |

## 显存行为

与 `vision_heavy` 的取向差异：

| | vision_fast（本节点） | vision_heavy |
|---|---|---|
| 加载时机 | **进程启动即加载** | 首次调用才加载 |
| 释放 | **从不自动释放** | 空闲 60s 后自动卸载 |
| 单次调用延迟 | 只有前向计算 | 冷启动需等加载 |
| 适合 | 高频、要低延迟 | 低频、模型偏重 |

启动时如果某个模型加载失败（例如部署机还没放权重），**节点不会起不来**：
错误会记在 `/health` 的 `detail.models[].error` 里，真正调用该工具时才报明确原因。

想强制卸载排查显存问题，可以调 `models.release_all()`。

## 接口

完全遵循 `docs/分工与接口约定.md` §4.2，与 `vision_heavy` 一致：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活 + 已注册工具名 + 模型/显存详情 |
| GET | `/tools` | 自报工具清单（网关按它发现工具） |
| POST | `/invoke` | 统一调用入口 |

工具内部报错时 **HTTP 仍为 200**，靠 `ok=false` + `error` 表达，不会拖垮节点。
