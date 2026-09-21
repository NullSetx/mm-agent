# mm-agent · 多模态系统

三个节点组成的多模态 AI 服务。分工、接口契约与协作规则见
[docs/分工与接口约定.md](../docs/分工与接口约定.md)——**接口以该文档 +
`common/schemas.py` 为唯一契约**。

| 节点 | 机器 | 目录 | 端口 | 职责 |
|------|------|------|------|------|
| 网关 | 16GB（A） | `llm_node/` | 8000 | FastAPI 网关 + LangChain Agent + vLLM(Qwen2-VL) |
| vision-fast | 8GB #1（B） | `vision_fast/` | 8101 | `detect` / `classify`，模型常驻 |
| vision-heavy | 8GB #2（C） | `vision_heavy/` | 8102 | `ocr` / `stylize`，懒加载 |

前端/用户只访问网关 `8000`；视觉节点只被网关调用。

## 快速开始

```bash
pip install -r requirements.txt        # 或 uv pip install -r requirements.txt

# 两台 8GB（B / C）
uvicorn vision_fast.server:app   --host 0.0.0.0 --port 8101
uvicorn vision_heavy.server:app  --host 0.0.0.0 --port 8102

# 16GB（A）网关
uvicorn llm_node.gateway:app     --host 0.0.0.0 --port 8000
```

跨机联调时注入 IP（不要写死在代码里）：

```bash
VISION_FAST_HOST=192.168.1.101 VISION_HEAVY_HOST=192.168.1.102 \
  uvicorn llm_node.gateway:app --host 0.0.0.0 --port 8000
```

PowerShell 语法：`$env:VISION_FAST_HOST="x.x.x.x"; ...`（注意 bash 的
`VAR=value cmd` 前缀写法在 PowerShell 里不识别）。

联调顺序（文档 §6.3）：各自起节点 → `GET /api/health` 全绿 →
`GET /api/tools` 聚合到工具 → `POST /api/invoke` 打通 → 接真实模型 →
最后 `POST /api/chat`。

## 浏览器测试台

网关自带一个测试页，直接浏览器打开：

```
http://<网关IP>:8000/
```

功能：上传图片（自动转 base64）、多轮对话（服务端记住会话）、工具调用
可视化（折叠面板看 JSON，stylize 等返回的图片直接内联显示）、右上角
各节点健康状态灯（15 秒自动刷新）。

## Agent 执行流程（带图请求）

```
用户消息 + 图片
  ↓ 网关预取（asyncio 并行，确定性执行、不依赖模型自觉）
detect + classify 分析图片；ocr 仅当问题带文字类意图时调用
（意图关键词门控：文字/写了/读到/号码…—— ocr 是预取里最慢的一环）
  ↓ render_for_llm() 按工具把 JSON 润色成中文观察
「[detect 观察] 共检测到 1 个物体。- person（置信度 0.91）…」
  ↓ 观察注入用户消息 → LangChain create_agent（≤5 轮）
LLM 依据观察回答（生成式工具如 stylize 由模型自主调用，结果图片内联展示）
```

设计要点：

- **观察文本是权威结果**：小模型读中文观察远比读嵌套 JSON 可靠，
  且杜绝编造——观察说没有就没有；
- **图片 base64 不进 LLM 上下文**（占位符替代），只走前端渲染通道；
- **新工具零接入成本**：没注册渲染器的工具走兜底渲染
  （精简 JSON，剔 `raw` 冗余副本、大字段占位）。

## 会话持久化

会话落盘 `data/sessions.db`（SQLite，标准库，已 gitignore），每轮对话
write-through 写回。**网关重启后，同一 `session_id` 继续对话，历史与
当前图片都在**——测试台的会话输入框填旧 ID 即可续聊。

- 历史保留最近 12 条，裁剪时对齐工具调用对（不会拦腰切断
  「AI 发起调用 → 工具观察」）；
- 用户消息只存文本进历史（图片每轮重新附带，观察已在工具消息里）；
- 库损坏/写失败自动降级为空会话并告警，不影响对话主流程；
- 会话数量无上限、无 TTL（课程项目规模，YAGNI）。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `VISION_FAST_HOST` / `VISION_HEAVY_HOST` | `127.0.0.1` | 两个视觉节点的 IP |
| `LLM_HOST` | `127.0.0.1` | vLLM 所在机器 IP |
| `GATEWAY_PORT` / `VLLM_PORT` / `VISION_FAST_PORT` / `VISION_HEAVY_PORT` | 8000 / 8001 / 8101 / 8102 | 端口 |
| `TOOL_TIMEOUT` / `CHAT_TIMEOUT` | 30 / 60 秒 | 单次工具调用 / 对话超时 |
| `NODE_MOCK` | 关 | `1` 时该节点返回合法占位结果（先 mock 后模型，文档 §6.2） |
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

## 加新工具（零改网关，文档 §5）

1. 在所属节点用 `common.registry.tool` 装饰器实现（四字段：name/description/needs_image/params）；
2. 重启该节点；
3. `POST /api/tools/refresh`。

`GET /api/tools` 出现它，Agent 自动多出一个能调用的工具。若新工具返回
结构特殊，可在 `llm_node/agent.py` 的 `_RENDERERS` 注册一个观察渲染函数
（不注册也能用，走兜底渲染）。

## 测试（网关侧）

```bash
pytest tests/
```

35 项，覆盖：工具发现/重名拒绝/404-502-504 错误映射、结果渲染层、
意图门控、会话持久化 round-trip、多轮上下文、mock 端到端。
测试不占端口：下游用 `common.node.build_app` 起真实契约的 ASGI app，
经进程内路由传输层连接；会话库写在 pytest 临时目录。

## 已知问题

- `common/config.py` 的 `base_url()`：对 `"llm"` 键会 KeyError、`"vllm"`/`"gateway"`
  键会静默错路由到 vision-heavy。网关侧已绕开（自拼 URL），修复需三人同意。
- WSL2 上 vLLM 默认禁用 pinned memory（`is_pin_memory_available()` 恒 False），
  启动会报 `UVA is not available`；本机 torch 锁页分配实测正常，属于保守开关，
  置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 即可（见上方环境变量表）。
- vLLM 0.26 的显存检查要求 空闲显存 ≥ `gpu-memory-utilization` × 总显存。
  Windows 桌面约占 1.1GB，所以 WSL 里 8GB 卡的该参数最高约 0.85。
- 3B 量化小模型偶尔无视"必须先调工具"的规则、凭自己的视觉直接回答，
  或对元问题（"我刚才问了什么"）答非所问——属模型能力上限，目标 7B
  明显更好；网关层的预取 + 观察注入已把影响压到最低。
- 系统代理会把发往 127.0.0.1 的请求代答成 502（连"服务不可达"的判断
  都会被带偏）。网关进程导入时已把 localhost/127.0.0.1 写入 `NO_PROXY`。
