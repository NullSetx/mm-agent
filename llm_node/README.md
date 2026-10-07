# 网关节点 · llm_node

**项目总览、一键启动、以及给自建前端的对接说明见根目录 [README.md](../README.md)。**
本文只讲网关侧细节：vLLM 部署、环境变量、对话接口、测试与已知问题。

分工、接口契约与协作规则见 [docs/分工与接口约定.md](../docs/分工与接口约定.md)
——**接口以该文档 + `common/schemas.py` 为唯一契约**。

| 节点 | 机器 | 目录 | 端口 | 职责 |
|------|------|------|------|------|
| 网关 | 16GB（A） | `llm_node/` | 8000 | FastAPI 网关 + LangChain Agent + vLLM(Qwen2-VL) |
| vision-fast | 8GB #1（B） | `vision_fast/` | 8101 | `detect` / `classify`，模型常驻 |
| vision-heavy | 8GB #2（C） | `vision_heavy/` | 8102 | `ocr` / `stylize`，懒加载 |

前端/用户只访问网关 `8000`；视觉节点只被网关调用。

## 快速开始

依赖唯一来源是根目录 `requirements.txt`（三人共享）。每台节点机器上装一套：

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

启动（各节点机器上，仓库根目录）：

```bash
# 8GB #1（B）
uvicorn vision_fast.server:app   --host 0.0.0.0 --port 8101

# 8GB #2（C）
uvicorn vision_heavy.server:app  --host 0.0.0.0 --port 8102

# 16GB（A）网关
uvicorn llm_node.gateway:app     --host 0.0.0.0 --port 8000
```

> ⚠️ **venv 别搞混**：网关和两个视觉节点共用仓库的 `.venv`；**vLLM 必须用自己
> 的 venv**（见下方「vLLM」）。两者依赖冲突，装不到一套里。
> 报 `No module named 'httpx'` / `'langchain'` / `'vllm'` 基本都是拿错了 venv。

### 单机跑通全链路（演示 / 开发用）

一张卡上跑四个进程：vLLM(8001) + 网关(8000) + 视觉节点(8101 / 8102)。
显存有限时**务必调低 `--gpu-memory-utilization`**，否则视觉节点会 OOM（见「vLLM」节）。

起完按这个顺序自检：

```bash
curl -s localhost:8000/api/health | python3 -m json.tool   # nodes 和 vllm 都该 ok
curl -s localhost:8000/api/tools  | python3 -m json.tool   # 应列出 detect/classify/ocr/stylize
```

`/api/health` 全绿即链路通。跨机联调时注入 IP（不要写死在代码里）：


```bash
VISION_FAST_HOST=192.168.1.101 VISION_HEAVY_HOST=192.168.1.102 \
  uvicorn llm_node.gateway:app --host 0.0.0.0 --port 8000
```

联调顺序（文档 §6.3）：各自起节点 → `GET /api/health` 全绿 →
`GET /api/tools` 聚合到工具 → `POST /api/invoke` 打通 → 接真实模型 →
最后 `POST /api/chat`。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `VISION_FAST_HOST` / `VISION_HEAVY_HOST` | `127.0.0.1` | 两个视觉节点的 IP |
| `LLM_HOST` | `127.0.0.1` | vLLM 所在机器 IP |
| `GATEWAY_PORT` / `VLLM_PORT` / `VISION_FAST_PORT` / `VISION_HEAVY_PORT` | 8000 / 8001 / 8101 / 8102 | 端口 |
| `TOOL_TIMEOUT` / `CHAT_TIMEOUT` | 30 / 60 秒 | 单次工具调用 / 对话超时 |
| `NODE_MOCK` | 关 | `1` 时该节点返回合法占位结果（先 mock 后模型，文档 §6.2） |
| `CORS_ALLOW_ORIGINS` | `*` | 允许跨域的前端来源，逗号分隔。前端与网关不同源，浏览器直连靠它；公网部署应改成具体来源 |
| `LLM_IMAGE_MAX_SIDE` | `1024` | 送给 LLM 的图片长边上限（像素）。**工具仍用原图**，只缩喂给模型的那份，见「图片与 token」 |
| `VLLM_MODEL_PATH` | `weights/Qwen2.5-VL-3B-Instruct-AWQ` | vLLM 加载的权重路径 |
| `VLLM_MODEL_NAME` | `qwen2.5-vl-3b-awq` | 对外模型名（OpenAI 接口的 model 字段） |
| `VLLM_MAX_MODEL_LEN` / `VLLM_GPU_MEMORY_UTILIZATION` | 4096 / 0.9 | vLLM 显存参数 |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | 关 | WSL2 上 vLLM 默认禁用锁页内存，本机实测可用，WSL 跑 vLLM 时置 `1` |

## vLLM（网关机器专用）

vLLM 只能跑 Linux：16GB 目标机原生跑；Windows 开发机走 WSL2。
安装（WSL 侧，Python 3.12）：

```bash
uv venv ~/vllm-venv --python 3.12
uv pip install --python ~/vllm-venv/bin/python vllm==0.26.0   # 不要升 0.27+，见 requirements.txt 注释
```

系统需 CUDA toolkit ≥ 12.9（flashinfer 现场编译 kernel 需要 nvcc；
CUDA 13.1 实测通过，只装 `cuda-toolkit-13-1`，别装会换驱动的 `cuda` 元包；
Ubuntu 26.04 的 multiverse 仓库直接有，WSL 里 `apt install cuda-toolkit-13-1` 即可）。

权重放 `weights/`（已 gitignore），从 ModelScope 下载（**要能自发调用工具，
最小只能用 3B**：2B 及以下的量化版不会输出 tool_call）：

```bash
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --local_dir weights/Qwen2.5-VL-3B-Instruct-AWQ
```

启动命令直接打印（在 WSL/Linux 侧执行）：

```bash
python -m llm_node.llm
# VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm serve weights/Qwen2.5-VL-3B-Instruct-AWQ \
#   --served-model-name qwen2.5-vl-3b-awq \
#   --chat-template llm_node/qwen25_tools_chat_template.jinja \
#   --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.9 \
#   --enable-auto-tool-choice --tool-call-parser hermes
```

`--chat-template` 不能省：VL 权重自带的模板不支持 tools，模型看不到工具
定义、永远不会自发调用（`qwen25_tools_chat_template.jinja` 来自 Qwen2.5
官方模板，支持 `<tool_call>` 格式）。

**16GB 机切 7B 只改路径，代码零改动**：
`VLLM_MODEL_PATH=weights/Qwen2-VL-7B-AWQ python -m llm_node.llm`。
8GB 显存紧张时降 `VLLM_GPU_MEMORY_UTILIZATION=0.8` 或 `VLLM_MAX_MODEL_LEN=2048`。

**单机演示要调低显存占用**：vLLM 默认 `0.9` 会先占走 14.4GB，但演示时是
LLM + 两个视觉节点**共用一张卡**，视觉节点会 OOM。实测 16GB 卡上用 **`0.5`**
（模型本身才 4GB 左右），四进程齐开后总占用约 12GB / 16GB。

### 换用 Qwen3-VL（可选）

官方 Qwen3-VL 没有 AWQ，只有 BF16/FP8/GGUF；社区量化版（如
`cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit`）实测在 vLLM 0.26.0 + sm_120 上可用。
两个坑：

- 必须 `--chat-template <模型目录>/chat_template.jinja`——该仓库的
  `tokenizer_config.json` **没有内嵌** chat_template，不指过去模型看不到工具定义。
  注意**不要**用仓库里那个 `qwen25_tools_chat_template.jinja`，那是 Qwen2.5 的。
- 起 vLLM 时 `--served-model-name` 要和网关的 `VLLM_MODEL_NAME` 一致，
  否则两边对不上（前端侧栏显示的模型名取自这里，对不上会看着像跑错了模型）。


## 对话接口（`POST /api/chat`）

契约见文档 §4.3。除契约字段外，网关还支持：

| 字段 | 默认 | 说明 |
|------|------|------|
| `stream` | `false` | `true` 时返回 **SSE** 逐段推送；不传则仍是原来的整段 JSON 响应（向后兼容） |

SSE 事件（每帧 `data: {...}`）：

| type | 载荷 | 说明 |
|------|------|------|
| `delta` | `text` | 回答的文本增量，一次一个 token |
| `tool` | `tool` / `ok` / `result` / `error` | 一条工具调用记录，**工具执行完即推送**，不必等整轮 |
| `done` | `reply` / `tool_calls` | 本轮结束，`reply` 为完整回答 |
| `error` | `message` | 生成过程中出错 |

另外每 10 秒可能收到 `: keepalive` 注释帧——那是保活用的，客户端忽略即可
（反代/隧道容易掐掉长时间静默的长连接）。

**图片只在「当轮上传」时参与分析**：用户传图那一轮，网关先并行调用
detect/classify/ocr，把结果润色成中文观察注入上下文（不赌小模型自觉调工具）；
之后的追问轮（如"谢谢你"）**不再重跑工具、也不再重复附图**，靠会话历史里那张图
和上一轮的分析结果。工具执行另走会话当前图，所以追问轮里模型仍能调 ocr/stylize。

**CORS**：前端是各人自己的展示页，与网关不同源，故网关已开跨域
（`CORS_ALLOW_ORIGINS`，默认放开）。前端可**直连** `http://<网关>:8000`，
不必再经自己的后端转发。

## 图片与 token

VL 模型的**视觉 token 数随分辨率平方增长**：一张 1600×2400 的网页截图约
**6916 token**，用 `--max-model-len 4096` 会被 vLLM 以 400 拒绝
（`Input length ... exceeds model's maximum context length`）。

因此喂给**模型**的那份图会先缩到长边 `LLM_IMAGE_MAX_SIDE`（默认 1024）。
工具（OCR / 检测）用的仍是**原图**，精度不受影响。实测缩完这段输入从 6916 降到
约 1500 token，预填也明显变快。该值调大会同时吃显存、变慢，一般不用动。

## 加新工具（零改网关，文档 §5）

1. 在所属节点用 `common.registry.tool` 装饰器实现（四字段：name/description/needs_image/params）；
2. 重启该节点；
3. `POST /api/tools/refresh`。

`GET /api/tools` 出现它，Agent 自动多出一个能调用的工具。

## 测试（网关侧）

```bash
pytest tests/
```

测试不占端口：下游用 `common.node.build_app` 起真实契约的 ASGI app，
经进程内路由传输层连接；`NODE_MOCK=1` 的 mock 端到端可在无模型环境跑。

## 已知问题

- `common/config.py` 的 `base_url()`：对 `"llm"` 键会 KeyError、`"vllm"`/`"gateway"`
  键会静默错路由到 vision-heavy。网关侧已绕开（自拼 URL），修复需三人同意。
- WSL2 上 vLLM 默认禁用 pinned memory（`is_pin_memory_available()` 恒 False），
  启动会报 `UVA is not available`；本机 torch 锁页分配实测正常，属于保守开关，
  置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 即可（见上方环境变量表）。
- vLLM 0.26 的显存检查要求 空闲显存 ≥ `gpu-memory-utilization` × 总显存。
  Windows 桌面约占 1.1GB，所以 WSL 里 8GB 卡的该参数最高约 0.85。
- **前端报 `Load failed`（Safari）/ `Failed to fetch`（Chrome）不等于网络问题**：
  这通常是网关侧抛了未捕获的异常，把 SSE 长连接掐断了，浏览器只能看到这句没有
  信息量的话。流式路径已加兜底 `except Exception`，会改发 `error` 事件并把完整
  栈写进网关日志——**先看网关日志，别先怀疑网络**。
