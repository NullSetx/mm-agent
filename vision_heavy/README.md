# vision_heavy 节点

`ocr` + `stylize` 两个工具。按文档分工跑在 **8GB #2** 节点上。
两个模型都是**懒加载 + 空闲超时自动释放显存**（见下方「显存行为」）。

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

## 冒烟测试

节点起来之后，跑一遍端到端测试：

```bash
python vision_heavy/smoke_test.py                 # 默认打 localhost:8102
python vision_heavy/smoke_test.py 192.168.1.102   # 打别的机器
```

它会自动造一张带中英文的测试图，依次调 `/health`、`/tools`、`ocr`、`stylize`，
打印识别结果和显存变化，结果图存到 `/tmp/vh_stylize_result.jpg`。

> 这个文件在 `.gitignore` 里，是本地自用的开发脚本。

## 契约自检

```bash
python -m common.selftest     # 18 项，不依赖模型权重，也不依赖队友
```

改动过 `common/` 或者怀疑自己的节点不合契约时跑一遍。

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

# 1) 主体文件走 ModelScope（快，约 4-5 MB/s）
B=https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6/resolve/master
for f in config.json model.safetensors tokenizer.json tokenizer.model \
         tokenizer_config.json added_tokens.json special_tokens_map.json \
         chat_template.jinja image_processing_paddleocr_vl.py \
         processing_paddleocr_vl.py configuration_paddleocr_vl.py \
         modeling_paddleocr_vl.py; do
  curl -sSL --retry 3 -O "$B/$f"
done

# 2) ModelScope 缺的三个配置从 HuggingFace 补（都是几百字节）
#    国内环境需要代理，换成你自己的端口
export https_proxy=http://127.0.0.1:<你的代理端口>
H=https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/resolve/main
for f in preprocessor_config.json processor_config.json generation_config.json; do
  curl -sSL -O "$H/$f"
done
```

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

**机制**：模型首次被调用时才加载，之后每次调用刷新「最后使用时间」；后台有个
回收线程每 5 秒检查一次，**空闲超过 `VISION_IDLE_TIMEOUT`（默认 60s）就卸载并归还显存**。

**注意不是「任一时刻只驻留一个」**：如果两个工具在一个空闲窗口内被连续调用，
两个模型会同时驻留。实测总占用约 **3.0 GB**，8GB 节点上仍有充足余量。
想强制只留一个，把 `VISION_IDLE_TIMEOUT` 调小即可。

实测数据（RTX 5060 Ti，384px 风格迁移）：

```
基线                    1375 MB
调 ocr 后               3271 MB    ← 加载 PaddleOCR-VL（约 +1.9GB）
调 stylize 后           3472 MB    ← 再加载 VGG19（约 +0.2GB）
空闲 75s 后             1084 MB    ← 两个都自动卸载，释放约 2.4GB
```

各模型的量级：

| 模型 | 参数量 | 增量显存 |
|---|---|---|
| PaddleOCR-VL-1.6 | 0.9B | 约 1.9 GB |
| VGG19（截断到 relu5_1） | 约 20M | 约 0.2 GB |

VGG19 之所以这么小，是因为**只保留了 relu5_1 之前的层**（后面的层风格迁移用不到），
且用 bf16 加载。风格迁移过程中会有额外的激活显存开销，随分辨率上升。

调 `models.release_all()` 可以立刻释放全部。

## 已知坑（实测踩过的）

**1. ModelScope 的 PaddleOCR-VL 镜像不全**

少了 `preprocessor_config.json` 会直接报：

```
OSError: Can't load image processor for '...'. ... containing a
preprocessor_config.json file
```

必须从 HuggingFace 补那三个配置文件，命令见上面的权重准备一节。

**2. `ALL_PROXY=socks://...` 会让模型下载崩掉**

huggingface_hub 用 httpx，而 httpx **不支持 `socks://` 协议**：

```
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')
```

如果你的环境里设了 `ALL_PROXY` 或 `all_proxy` 指向 socks，跑之前先清掉，
只保留 http/https 的代理：

```bash
unset ALL_PROXY all_proxy
export https_proxy=http://127.0.0.1:<端口> http_proxy=http://127.0.0.1:<端口>
```

**3. 部署机显卡未知，设备要能自动降级**

代码里不假设任何具体 GPU：优先 CUDA、失败退 CPU，精度按计算能力在
`bf16 → fp16 → fp32` 之间选。想强制指定用 `VISION_DEVICE`。
CPU 上功能正常，只是慢（128px 风格迁移约 0.5s，512px 就要几十秒）。
