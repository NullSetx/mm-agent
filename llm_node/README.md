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
detect + classify 分析图片作保底；读字（ocr）、生成（stylize）不预取，
由模型按系统提示的决策策略自主调用（问题需要读字 → 调 ocr）
  ↓ render_for_llm() 按工具把 JSON 润色成中文观察
「[detect 观察] 共检测到 1 个物体。- person（置信度 0.91）…」
  ↓ 观察注入用户消息 → LangChain create_agent（≤5 轮）
LLM 依据观察回答（生成式工具如 stylize 由模型自主调用，结果图片内联展示）
```

### 附件分流（PDF / 文本走另一条路）

`/api/chat` 的 `image` 字段按 `data:` 前缀里声明的 mime 分两路
（见 `split_attachment`）。**没有前缀一律按图片处理**——A 的测试台和历史客户端
发的都是裸 base64，这个语义不能改。

```
文档（application/pdf、text/*、代码文件）
  ↓ 网关调 vision-heavy 的 read_document，读成正文
  ↓ 正文拼进用户消息（带「【附带文件：…】」抬头），随历史留存 → 追问接得上
  ↓ 既不附进多模态 parts（把 PDF 当 image_url 发给 vLLM 直接 400）
  ↓ 也不存进 session.image（那是"当前图片"，detect / ocr 拿到 PDF 只会报错）
```

`read_document` 的参数名以下划线开头，`args_schema` 会**滤掉不暴露给模型**，值只能
由网关注入——模型不可能知道一份 PDF 的 base64。不滤的后果实测过：4B 会自己瞎编参数
去调，拿到 base64 报错后**照着报错回答**，把已经放进上下文的正文全无视掉。网关那边
另外无条件覆盖一遍参数，兜住模型自发调用的情况。

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

- **两道裁剪闸并行，谁先到算谁**：条数（`MAX_HISTORY_MESSAGES`，默认 120）和
  **字符预算**（`MAX_HISTORY_CHARS`，默认按**探测到的** vLLM 实际窗口减 3500 推导）。
  只按条数管不住上下文——一条带附件正文的用户消息可能几千字，十几条就把窗口撑爆，
  vLLM 直接回 400 把整轮对话打断，而那条报错用户完全无从下手。超预算从最旧的丢，
  **最新那条一定保留**（那是用户刚说的话，丢了只会答非所问）；
- 裁剪对齐工具调用对（不会拦腰切断「AI 发起调用 → 工具观察」）；
- **发送前也要裁一遍**，不能只在轮末裁：库里存的历史可能是上一次配置留下的
  （比如旧的 120 条上限），这一轮会原样发出去、照样 400（实测踩过）；
- 用户消息只存文本进历史（图片每轮重新附带，观察已在工具消息里）；
  **附件正文会拼进用户消息一起留存**，追问「第 3 页说了什么」才接得上；会话回放
  时 `_history_view` 会按标记把正文切掉，界面上只显示用户真正打的那句话；
- ⚠️ **流式和一次性两条路径都必须落盘**。流式那条曾经漏了保存——端点函数随
  `return StreamingResponse(...)` 就退出了，轮不到它调 `sessions.save()`，结果
  **所有走 SSE 的前端每轮都是空会话**（追问永远接不上），而 SSE 的 `done` 事件照样
  带着 `history_len`，界面上完全看不出来。已修，并有回归测试守着；
- 库损坏/写失败自动降级为空会话并告警，不影响对话主流程；
- 会话数量无上限、无 TTL（课程项目规模，YAGNI）。

## 上下文预算与采样

两个**实测调出来、别凭感觉改**的参数：

**`LLM_TEMPERATURE` 千万别设 0。** 贪心解码下，只要上下文里出现过重复内容，模型
"最可能的下一个 token"就是继续复读它——自我强化，一旦开始停不下来，而且每轮回复
都进历史，等于每轮给自己加码。同一会话连发 6 轮「喵」实测：

| 配置 | 相邻两轮回复的相似度 |
|---|---|
| `temperature=0.0` | 0.24 → 0.32 → 0.44 → **0.86 → 0.96**（收敛成复读） |
| `temperature=0.4` | 0.24 → 0.32 → 0.42 → **0.86 → 0.96**（一模一样，白改） |
| `temperature=0.7` + `presence_penalty=0.8` | 0.20 → 0.36 → 0.28 → 0.31 → 0.27（一直是新内容） |

复读的主因不是随机性，是**上下文**；0.4 这种"折中"毫无用处，只有模型作者本来配的
0.7 才压得住。权重自带的 `generation_config.json` 就是 `0.7 / top_p 0.8 / top_k 20`。

**`history_char_budget()` 的窗口取探测值，不取 `.env`。** 网关启动时和每次
`/api/health` 都探一次 vLLM（`GET /v1/models` 里的 `max_model_len`）。原因：改了
`.env` 却没重启 vLLM 时两者会不一致——实测 `.env` 写 102400、跑着的 VLLM 还是
10240，按 `.env` 算出的预算（98900）等于白设。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `VISION_FAST_HOST` / `VISION_HEAVY_HOST` | `127.0.0.1` | 两个视觉节点的 IP |
| `KB_HOST` / `KB_PORT` | `127.0.0.1` / `8103` | kb 知识库节点（见 [kb_node/README.md](../kb_node/README.md)） |
| `LLM_HOST` | `127.0.0.1` | vLLM 所在机器 IP |
| `GATEWAY_PORT` / `VLLM_PORT` / `VISION_FAST_PORT` / `VISION_HEAVY_PORT` / `KB_PORT` | 8000 / 8001 / 8101 / 8102 / 8103 | 端口 |
| `TOOL_TIMEOUT` / `CHAT_TIMEOUT` | 30 / 60 秒 | 单次工具调用 / 对话超时 |
| `NODE_MOCK` | 关 | `1` 时该节点返回合法占位结果（先 mock 后模型，文档 §6.2） |
| `VLLM_MODEL_PATH` | `weights/Qwen2.5-VL-3B-Instruct-AWQ` | vLLM 加载的权重路径 |
| `VLLM_MODEL_NAME` | `qwen2.5-vl-3b-awq` | 对外模型名（OpenAI 接口的 model 字段） |
| `VLLM_MAX_MODEL_LEN` / `VLLM_GPU_MEMORY_UTILIZATION` | 4096 / 0.9 | vLLM 显存参数。**窗口受显存限制**，填超了 vLLM 起不来（报错里会给它估的上限） |
| `LLM_TEMPERATURE` / `LLM_TOP_P` / `LLM_PRESENCE_PENALTY` | 0.7 / 0.8 / 0.8 | 采样参数，**别把 temperature 设 0**（见上文「上下文预算与采样」） |
| `MAX_HISTORY_MESSAGES` | 120 | 历史条数上限，与 `MAX_HISTORY_CHARS` 并行、谁先到算谁 |
| `MAX_HISTORY_CHARS` | 自动 | 历史字符预算，默认按**探测到的** vLLM 实际窗口减 3500 |
| `LLM_IMAGE_MAX_SIDE` | 1024 | 只影响**送给模型看的**那份图的长边；工具拿到的始终是原图 |
| `CORS_ALLOW_ORIGINS` | `*` | 前端来源固定时收紧成逗号分隔的完整来源 |
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
