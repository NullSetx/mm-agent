# web —— 演示前端（成员 B 交付）

校园网内用浏览器打开就能演示的智能体页面。页面**只调网关的 `/api/*`**（文档 §1/§4.3），
模型调用全部走 HTTP 端口；这里额外做了一层**同源代理**，所以浏览器不需要任何 CORS 配置，
也不用改 `llm_node/`（A 的目录）。

```
浏览器 ──同源──▶ web 前端 :8103 ──代理 /api/*──▶ 网关 :8000 ──▶ vision-fast :8101 / vision-heavy :8102
                                                       └──────▶ vLLM :8001
```

## 启动

在**仓库根目录**、用项目环境（`D:\Anaconda\Anaconda\envs\mm-agent`）：

```bash
# 本机（B，8GB #1）：视觉节点 + 演示前端
uvicorn vision_fast.server:app --host 0.0.0.0 --port 8101
uvicorn web.server:app      --host 0.0.0.0 --port 8103

# 演示时网关跑在 A 的机器上（由 A 启动）
uvicorn llm_node.gateway:app --host 0.0.0.0 --port 8000
```

没有权重 / 没有 vLLM 时，用 mock 模式先把页面跑起来（文档 §6.2）：

```bash
NODE_MOCK=1 uvicorn vision_fast.server:app --host 0.0.0.0 --port 8101
NODE_MOCK=1 uvicorn llm_node.gateway:app   --host 0.0.0.0 --port 8000
uvicorn web.server:app                     --host 0.0.0.0 --port 8103
```

打开页面：本机 `http://127.0.0.1:8103`；同网段同学用 `http://<你的局域网IP>:8103`
（启动日志会直接打印可用地址）。

## 端口与环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `WEB_HOST` | `0.0.0.0` | 前端监听地址，**必须 0.0.0.0** 才能被同网段访问（文档 §4.1） |
| `WEB_PORT` | `8103` | 前端端口。契约里的端口是 8000/8001/8101/8102，8103 为本前端新增（不改 `common/`，已在群里报备） |
| `LLM_HOST` | `127.0.0.1` | **网关所在机器**。演示时填 A 机器的局域网 IP |
| `GATEWAY_PORT` | `8000` | 网关端口（契约） |
| `GATEWAY_URL` | 空 | 直接指定网关基址，优先级高于 `LLM_HOST`+`GATEWAY_PORT` |
| 超时 | 契约值 | 代理 `/api/chat` 用 `CHAT_TIMEOUT`(60s)，其余用 `TOOL_TIMEOUT`(30s)，取自 `common.config` |

示例：网关在 A 机上（`192.168.1.20`）时启动前端

```bash
LLM_HOST=192.168.1.20 GATEWAY_PORT=8000 uvicorn web.server:app --host 0.0.0.0 --port 8103
```

### 代理目标是动态的

页面右上角有**「网关地址」输入框**，填 `192.168.1.20:8000` 点「应用」即刻生效
（浏览器把目标放进请求头 `X-Gateway-Target`，服务端**每个请求都重新解析**，无需重启）。
解析优先级：**请求头 → `GATEWAY_URL` → `LLM_HOST`+`GATEWAY_PORT`**。
演示当天 A 的 IP 变了，现场改一下就行。

## 页面功能

| 区域 | 用的接口 | 说明 |
|---|---|---|
| 节点状态 | `GET /api/health`、`POST /api/tools/refresh` | 网关 / vLLM / vision-fast / vision-heavy 四张卡，绿红指示 + 延迟 + 已注册工具；改地址后可点「刷新状态」验证连通 |
| 当前图片 | — | 对话与工具直调共用一张图；前端压缩到长边 ≤1280、JPEG 0.85 再转 base64，避免请求体过大 |
| 图文对话 | `POST /api/chat` | 同一会话连续提问（`session_id` 存在浏览器本地）；回复下方列出**实际发生的工具调用**及其结果摘要 |
| 工具直调 | `GET /api/tools`、`POST /api/invoke` | 工具下拉（自动包含 C 的 `ocr`/`stylize`）；参数表单按 `params` 默认值类型自动生成；结果可视化：detect 在图上画框、classify 画 top-k 条形图、ocr 列文本、stylize 显示返回图；附节点耗时与端到端耗时、原始 JSON 折叠显示 |
| 一键自检 | 同上 | 依次跑 `detect` + `classify`，输出耗时与结果摘要，现场 10 秒证明 B 节点活着 |

## 自检

```bash
python web/smoke_test.py          # 一条命令起 mock 三件套并走代理验证全链路，跑完自动关掉
```

覆盖：静态页与静态资源、`/__config`、代理透传 `health`/`tools`/`tools/refresh`/`invoke`/`chat`、
字符串参数强转、未知工具 404 透传、**动态目标**（请求头指向错误地址应得 502、去掉后恢复）。

端口被占时：`WEB_PORT=18103 GATEWAY_PORT=18000 VISION_FAST_PORT=18101 python web/smoke_test.py`

## 校园网演示注意事项

1. **Windows 防火墙**：首次启动会弹「是否允许」，选**专用网络**；或用管理员命令放行（每台机开自己的端口）
   ```bat
   netsh advfirewall firewall add rule name="mm-agent 8103" dir=in action=allow protocol=TCP localport=8103
   ```
   A 的机器放行 8000，C 的机器放行 8102，B 的机器放行 8101 与 8103。
2. **校园网可能有 AP 隔离**：连着同一个 SSID 也可能互相 ping 不通。演示前先用手机开热点、或把三台机器接到同一台交换机/路由器上自测；页面「节点状态」能直接反映谁没通。
3. **无外网也能用**：页面零 CDN 依赖，所有资源本地加载。
4. **没有鉴权**（按约定不做口令）：同网段任何人打开地址都能用，仅适合演示场景。
5. 手机浏览器同样可访问，建议把 `http://<B机IP>:8103` 生成二维码方便同学扫码体验。

## 契约与边界

- 只调 `/api/*`（文档 §4.3）；**不直连** 8101/8102 节点，也不在前端引入任何模型代码。
- **不修改** `common/`、`llm_node/`、`vision_heavy/`；本目录是新增的独立目录。
- 网关地址一律从环境变量/页面取值，代码里不写死 IP（文档 §4.1）。
