# vision_heavy 节点

`ocr` + `stylize` 两个工具。按文档分工跑在 **8GB #2** 节点上。
两个模型都**懒加载 + 空闲自动释放显存**，任一时刻最多驻留一个。

## 启动

```bash
# 从仓库根目录
pip install -r requirements.txt

# 正常启动
uvicorn vision_heavy.server:app --host 0.0.0.0 --port 8102

# 没有模型权重时，用 mock 模式先把链路跑通（文档 §6.2）
NODE_MOCK=1 uvicorn vision_heavy.server:app --host 0.0.0.0 --port 8102
```

绑定 `0.0.0.0` 才能被其他机器的网关访问（文档 §4.1）。

## 自检

```bash
python -m common.selftest     # 契约自检，不需要模型权重，18 项
curl localhost:8102/health    # 看设备、显存、已加载模型
curl localhost:8102/tools     # 看工具清单
```

## 权重准备

模型权重不进仓库（`.gitignore` 覆盖）。两个模型的获取方式不同：

### stylize 用的 VGG19

由 `torchvision` 自动下载到 `~/.cache/torch/hub/checkpoints/`，**无需手动操作**。
首次调用 `stylize` 时会自动拉取（约 574MB）。

### ocr 用的 PaddleOCR-VL

权重约 1.8GB，按下面的顺序查找：

1. 环境变量 `OCR_MODEL_PATH` 指向的目录
2. `~/.cache/mm-agent/models/PaddleOCR-VL-1.6/`
3. 都没有才联网拉取（HuggingFace / ModelScope）

**推荐预先下载到本地**，这样部署机的节点可以完全离线启动。

国内环境从 ModelScope 取速度快（直连即可，实测 4-5 MB/s），但**它的镜像不完整**，
缺三个 transformers 必需的配置文件，必须再从 HuggingFace 补：

```bash
D=~/.cache/mm-agent/models/PaddleOCR-VL-1.6
mkdir -p "$D" && cd "$D"

# 1) 主体文件走 ModelScope（快）
B=https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6/resolve/master
for f in config.json model.safetensors tokenizer.json tokenizer.model \
         tokenizer_config.json added_tokens.json special_tokens_map.json \
         chat_template.jinja image_processing_paddleocr_vl.py \
         processing_paddleocr_vl.py configuration_paddleocr_vl.py \
         modeling_paddleocr_vl.py; do
  curl -sSL --retry 3 -O "$B/$f"
done

# 2) ModelScope 缺的三个配置从 HuggingFace 补（都是几百字节）
#    国内环境需要代理，换成你自己的
export https_proxy=http://127.0.0.1:<你的代理端口>
H=https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/resolve/main
for f in preprocessor_config.json processor_config.json generation_config.json; do
  curl -sSL -O "$H/$f"
done
```

> 少了 `preprocessor_config.json` 会直接报
> `Can't load image processor ... containing a preprocessor_config.json file`，
> 这是实测踩过的坑。

> 如果部署机没有可用的 CUDA，或 PaddleOCR-VL 在目标显卡上跑不起来，
> 可以改用 **PP-OCRv6**（34.5M 参数，纯 CPU 也能跑，几乎不占显存）。
> transformers 5.x 同样内置了 `PPOCRV6TinyRec` / `PPOCRV6SmallRec` 等实现，
> 用 `OCR_MODEL_ID` 指过去即可，工具接口完全不变。

### 风格图

`stylize` 的风格来自 `vision_heavy/data/styles/` 下的图片，**文件名即风格名**。
把这个目录填上，工具描述里就会自动列出可用风格供 LLM 选择。

目录被 `.gitignore` 覆盖，各人本地准备。当前已有的 5 幅均为公有领域作品
（梵高《星月夜》《向日葵》、葛饰北斋《神奈川冲浪里》、蒙克《呐喊》、
修拉《大碗岛的星期天下午》）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `NODE_MOCK` | 空 | 置 `1` 时工具返回占位结果，用于无权重联调 |
| `VISION_DEVICE` | 自动 | 强制设备，如 `cpu` / `cuda:0`。不设则优先 CUDA、失败退 CPU |
| `VISION_IDLE_TIMEOUT` | `60` | 模型空闲多少秒后卸载显存；设 `0` 表示从不自动卸载 |
| `OCR_MODEL_PATH` | 空 | 本地 OCR 权重目录，优先级最高 |
| `OCR_MODEL_ID` | `PaddlePaddle/PaddleOCR-VL-1.6` | 联网拉取时用的模型 id |
| `VISION_HEAVY_HOST` | `127.0.0.1` | 供网关发现本节点（`common/config.py` 读取） |

## 显存行为

懒加载的实际效果（384px 风格迁移实测）：

```
调用前          1224 MB
调用后          2499 MB    ← 加载 VGG19
空闲 75s 后     1624 MB    ← 自动卸载
```

想立刻释放可以调 `models.release_all()`，或把 `VISION_IDLE_TIMEOUT` 调小。
