# mm-agent · 多模态视觉智能体

把「目标检测 / 图像分类 / OCR / 风格迁移」四个视觉能力包成统一的 HTTP 服务，
再由一个大模型 Agent 根据用户的自然语言提问**自主决定调哪个工具**。

前端只需要对着一个网关发请求：上传图片 + 提问题，网关负责调工具、把结果喂给
模型、把回答流式吐回来。**前端不必知道背后有几个节点、工具怎么调。**

```
                        你的前端（任意语言 / 框架）
                                 │  HTTP + CORS
                                 ▼
        ┌──────────────────────────────────────────────┐
        │  网关  llm_node/           :8000              │
        │  /api/health  /api/tools  /api/chat(SSE)      │
        │  大模型 Agent（vLLM，OpenAI 兼容）      :8001  │
        └───────┬──────────────────────────┬───────────┘
                │ 工具调用                  │ 工具调用
                ▼                          ▼
      ┌────────────────────┐    ┌────────────────────┐
      │ vision_fast  :8101 │    │ vision_heavy :8102 │
      │ detect / classify  │    │ ocr / stylize      │
      │ 模型常驻显存        │    │ 懒加载 + 空闲释放    │
      └────────────────────┘    └────────────────────┘
```

| 节点 | 目录 | 端口 | 职责 | 权重策略 |
|---|---|---|---|---|
| 网关 | `llm_node/` | 8000 | FastAPI 网关 + LangChain Agent + vLLM | — |
| vision-fast | `vision_fast/` | 8101 | `detect`（YOLO）/ `classify`（ResNet50） | 常驻显存 |
| vision-heavy | `vision_heavy/` | 8102 | `ocr`（PaddleOCR-VL）/ `stylize`（Gatys+VGG19） | 懒加载，空闲释放 |

**加新工具不用改网关**：在所属节点用 `@tool` 声明四个字段、重启该节点、
`POST /api/tools/refresh` 即可，Agent 会自动多出一个能调的工具。

## 快速开始

### 一键脚本（推荐）

```bash
git clone <本仓库> && cd mm-agent
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

./scripts/mm-agent.sh start --mock      # 先跑 mock：不需要任何模型权重，秒起
```

`--mock` 模式下所有工具返回结构合法的占位结果，**前端开发时用这个最省事**——
不用等权重下载、不占显存。

跑真实模型：

```bash
# 网关 + 两个视觉节点 + vLLM
./scripts/mm-agent.sh start --with-vllm --model weights/Qwen3-VL-4B-Instruct-AWQ-4bit

./scripts/mm-agent.sh status              # 看谁在跑
./scripts/mm-agent.sh stop
```

`--model` 指向 `weights/` 下的权重目录即可，**服务名和 chat template 会自动推导**：

- 服务名 = 目录名小写，并且**同一个名字也喂给网关**，所以两边不可能对不上
- chat template 优先用**该模型目录自带的** `chat_template.jinja`（Qwen3-VL 必须用
  自带的；仓库里那份是 Qwen2.5 专用，混用会让模型看不到工具定义）

不传 `--model` 时用 `VLLM_MODEL_PATH`，再没有就退回 Qwen2.5 的默认路径。

> vLLM 在**另一个 venv**里（它和仓库 `.venv` 依赖冲突，装不到一起）。
> 脚本默认找 `~/vllm-venv`，用 `VLLM_VENV=/path/to/venv` 覆盖。详见
> [llm_node/README.md](llm_node/README.md#vllm网关机器专用)。

### 手动起

```bash
# 8GB #1（B）
uvicorn vision_fast.server:app   --host 0.0.0.0 --port 8101
# 8GB #2（C）
uvicorn vision_heavy.server:app  --host 0.0.0.0 --port 8102
# 16GB（A）网关
uvicorn llm_node.gateway:app     --host 0.0.0.0 --port 8000
```

跨机联调时用环境变量注入各节点 IP（**不要写死在代码里**）：

```bash
VISION_FAST_HOST=192.168.1.101 VISION_HEAVY_HOST=192.168.1.102 \
  uvicorn llm_node.gateway:app --host 0.0.0.0 --port 8000
```

## 前端怎么接

前端由各人自己写，**不需要写后端代理**。网关已开 CORS，浏览器可以直接连。

### 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 聚合健康状态。**联调先看它**，一眼看出哪个节点没起 |
| GET | `/api/tools` | 工具清单（名字 / 说明 / 参数 / 是否需要图片） |
| POST | `/api/tools/refresh` | 重新发现工具（节点加了新工具后调） |
| POST | `/api/invoke` | 直接调**某一个**工具，不经大模型 |
| POST | `/api/chat` | **对话主入口**，可带图，支持 SSE 流式 |

### 推荐连法：浏览器直连网关

```js
const GATEWAY = 'http://192.168.1.100:8000';   // 网关所在机器，浏览器要能访问到
```

⚠️ **别写死 `127.0.0.1:8000`**。浏览器里的 `127.0.0.1` 指的是**用户自己的机器**，
不是服务器。前端和网关不在同一台机器时，这会让请求打到用户自己的电脑上。
建议按当前访问的主机名推导，或用配置项：

```js
const GATEWAY = `${location.protocol}//${location.hostname}:8000`;
```

CORS 默认全放开（`CORS_ALLOW_ORIGINS`）；要收紧就把它设成具体来源，逗号分隔：

```bash
CORS_ALLOW_ORIGINS=http://192.168.1.50:8080,http://localhost:3000 \
  uvicorn llm_node.gateway:app --port 8000
```

### 对话：`POST /api/chat`

请求：

```json
{
  "session_id": "abc",              // 前端生成并保持，同 id 多轮续聊
  "message": "图里有什么？",
  "image": "data:image/png;base64,...",   // 可选，base64，可带 data: 前缀
  "stream": true                    // 可选，true 走 SSE；不传则一次性返回 JSON
}
```

`stream: false`（默认）时返回：

```json
{ "reply": "图中有 2 个目标……", "tool_calls": [{"tool": "detect", "ok": true, "result": {}}] }
```

`stream: true` 时响应是 **SSE**（`text/event-stream`），逐帧推送：

| type | 载荷 | 说明 |
|---|---|---|
| `delta` | `text` | 回答的文本增量，一次一个 token |
| `tool` | `tool` / `ok` / `result` / `error` | 工具调用记录，**跑完即推**，不用等整轮 |
| `done` | `reply` / `tool_calls` | 本轮结束，`reply` 是完整回答 |
| `error` | `message` | 生成中出错 |

还会收到 `: keepalive` 之类的**注释帧**——那是保活用的，解析时忽略即可
（反代容易掐掉长时间静默的连接）。

命令行看一眼：

```bash
curl -N -X POST http://192.168.1.100:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"你好","stream":true}'
```

浏览器里读（**不能用 `EventSource`**，它只支持 GET；用 `fetch` + 流）：

```js
const res = await fetch(GATEWAY + '/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: sid, message, image, stream: true }),
});

const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = '';
for (;;) {
  const { done, value } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  let cut;
  while ((cut = buf.indexOf('\n\n')) >= 0) {      // SSE 以空行分帧
    const frame = buf.slice(0, cut); buf = buf.slice(cut + 2);
    const line = frame.split('\n').find((l) => l.startsWith('data:'));
    if (!line) continue;                          // 注释帧，跳过
    const ev = JSON.parse(line.slice(5));
    if (ev.type === 'delta')      append(ev.text);
    else if (ev.type === 'tool')  showToolCard(ev);
    else if (ev.type === 'done')  finish(ev.reply);
    else if (ev.type === 'error') fail(ev.message);
  }
}
```

### 两个行为约定

1. **图片只在「上传的那一轮」参与分析。** 传图那轮网关会先并行调
   `detect`/`classify`/`ocr`，把结果喂给模型；之后的追问轮（如"谢谢你"）
   **不会重跑工具**，靠会话历史回答。所以前端一轮里只需要在用户真正选了图时
   带上 `image`。
2. **图片别传太大。** 服务端会把喂给模型的那份缩到长边 1024，但 base64 走网络
   仍占带宽。前端压缩到长边 ~1024 再传最省事。

### 只想调单个工具？

不想走大模型，就直接打 `/api/invoke`：

```bash
curl -X POST http://192.168.1.100:8000/api/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool":"detect","image":"<base64>","params":{"conf":0.3}}'
```

响应固定形状（**工具内部报错也是 HTTP 200**，靠 `ok:false` 表达）：

```json
{ "ok": true, "tool": "detect", "result": {"boxes": [], "count": 0}, "error": null, "elapsed_ms": 12.3 }
```

只有**未知工具**才返回 404；下游节点挂了 502、超时 504。

## 环境变量

常用几个（完整表见 [llm_node/README.md](llm_node/README.md#环境变量)）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `VISION_FAST_HOST` / `VISION_HEAVY_HOST` | `127.0.0.1` | 两个视觉节点 IP |
| `LLM_HOST` | `127.0.0.1` | vLLM 所在机器 |
| `NODE_MOCK` | 关 | `1` = 返回占位结果，不加载模型 |
| `CORS_ALLOW_ORIGINS` | `*` | 允许跨域的前端来源 |
| `LLM_IMAGE_MAX_SIDE` | `1024` | 喂给模型的图片长边上限 |
| `VLLM_MODEL_NAME` | `qwen2.5-vl-3b-awq` | **要和 vLLM 的 `--served-model-name` 一致** |

## 常见问题

- **报 `No module named 'httpx'` / `'langchain'` / `'vllm'`** —— 用错 venv 了。
  网关和视觉节点用仓库 `.venv`；vLLM 用它自己的 venv。
- **前端报 `Load failed`（Safari）/ `Failed to fetch`（Chrome）** —— 多半不是网络
  问题，而是网关侧抛了异常把 SSE 连接掐断了。**先看网关日志**（`./scripts/mm-agent.sh`
  起的在 `.run/logs/gateway.log`）。
- **对话报「LLM 服务不可达」** —— vLLM 没起。只想看界面就加 `--mock`。
- **报 `Input length ... exceeds model's maximum context length`** —— 图片太大。
  服务端已做缩图，若仍出现说明调大了 `LLM_IMAGE_MAX_SIDE` 或把
  `--max-model-len` 设小了。

## 相关文档

- [docs/分工与接口约定.md](docs/分工与接口约定.md) —— **接口契约唯一依据**，改动需三人同意
- [llm_node/README.md](llm_node/README.md) —— 网关与 vLLM 细节、环境变量全表
- [vision_fast/README.md](vision_fast/README.md) / [vision_heavy/README.md](vision_heavy/README.md) —— 两个视觉节点
- `scripts/mm-agent.sh --help` —— 一键脚本

## 后端契约自检

```bash
python -m common.selftest     # 不依赖模型权重
pytest tests/ -p anyio        # 网关侧测试
```
