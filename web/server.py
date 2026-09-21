"""web 演示前端：托管静态页 + 把 `/api/*` 同源代理到网关。

为什么需要这一层（而不是让页面直连网关）：
- 文档 §1 规定「前端只访问网关」，而网关（`llm_node/gateway.py`）**没有开 CORS**。
  页面若在 :8103 直接 `fetch` 网关的 :8000，会被浏览器按跨域拦下。
  在这里做同源代理后，浏览器只与本服务打交道：**零 CORS 配置、零改动 A 的目录**。
- 所有模型调用一律走 HTTP 端口（页面 → 本服务 → 网关 → 各节点），前端不引入任何
  模型代码，也不直连 8101/8102（契约 §4.3：前端只认 `/api/*`）。

代理目标是**动态**的，按请求逐个解析，优先级：
1. 请求头 `X-Gateway-Target` —— 页面上可直接改，演示当天换 IP 不用重启；
2. 环境变量 `GATEWAY_URL`；
3. `common.config` 的 `LLM_HOST` + `GATEWAY_PORT`（默认 127.0.0.1:8000）。

端口与超时同样按契约风格从配置取：
- 本服务：`WEB_HOST`（默认 0.0.0.0，必须绑 0.0.0.0 才能被同网段访问）/ `WEB_PORT`（默认 8103）
- 代理超时：`/api/chat` 用 `CHAT_TIMEOUT`(60s)，其余用 `TOOL_TIMEOUT`(30s)（文档 §4.1）

启动：

    uvicorn web.server:app --host 0.0.0.0 --port 8103

只读诊断：

    GET /healthz    本服务自身存活
    GET /__config   当前生效的网关地址与超时
"""

from __future__ import annotations

import logging
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from common.config import CHAT_TIMEOUT, HOSTS, PORTS, TOOL_TIMEOUT

log = logging.getLogger("web.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: 代理转发时连接阶段单独设短超时，避免网关没起时页面长时间转圈
CONNECT_TIMEOUT = 5.0

#: 网关节点的配置键（common.config 里 llm = 网关所在机器）
GATEWAY_HOST_KEY = "llm"
GATEWAY_PORT_KEY = "gateway"


def web_host() -> str:
    return os.getenv("WEB_HOST", "0.0.0.0").strip() or "0.0.0.0"


def web_port() -> int:
    return int(os.getenv("WEB_PORT", "8103"))


def normalize_target(raw: str) -> str:
    """把用户填的地址规范成 `http://host:port`（容忍省略协议、末尾斜杠、带 /api 后缀）。"""
    text = (raw or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/api"):
        text = text[: -len("/api")]
    if not text.startswith(("http://", "https://")):
        text = "http://" + text
    return text


def resolve_gateway(request: Request | None = None) -> str:
    """动态解析网关基址：请求头 → 环境变量 → common.config 默认值。

    每次都重新解析（不缓存），所以页面上改地址后**后续请求立即生效**，无需重启。
    """
    if request is not None:
        header = normalize_target(request.headers.get("x-gateway-target", ""))
        if header:
            return header

    url = normalize_target(os.getenv("GATEWAY_URL", ""))
    if url:
        return url

    host = os.getenv("LLM_HOST", HOSTS.get(GATEWAY_HOST_KEY, "127.0.0.1"))
    port = os.getenv("GATEWAY_PORT", str(PORTS.get(GATEWAY_PORT_KEY, 8000)))
    return f"http://{host}:{port}"


def local_ipv4() -> list[str]:
    """本机局域网 IPv4，启动时打印，方便把访问地址告诉同学。"""
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:  # 兜底：UDP socket 不发包，只为拿到出口网卡地址
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        found.add(sock.getsockname()[0])
        sock.close()
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith("127."))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(follow_redirects=True)
    port = web_port()
    log.info("演示前端已启动：本机 http://127.0.0.1:%s", port)
    for ip in local_ipv4():
        log.info("同网段可访问：http://%s:%s", ip, port)
    log.info("当前网关：%s（页面顶部可改，或设 GATEWAY_URL / LLM_HOST / GATEWAY_PORT）",
             resolve_gateway())
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="mm-agent demo web", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------- 自身端点


@app.get("/healthz")
async def healthz() -> dict:
    """本服务存活（不依赖网关）。"""
    return {
        "ok": True,
        "service": "web",
        "web_port": web_port(),
        "gateway": resolve_gateway(),
    }


@app.get("/__config")
async def config(request: Request) -> dict:
    """当前生效的配置，页面顶部据此显示「正在指向哪个网关」。"""
    return {
        "gateway": resolve_gateway(request),
        "fallback_gateway": resolve_gateway(None),
        "web_host": web_host(),
        "web_port": web_port(),
        "timeouts": {"chat": CHAT_TIMEOUT, "tool": TOOL_TIMEOUT},
        "local_urls": [f"http://{ip}:{web_port()}" for ip in local_ipv4()],
    }


# ---------------------------------------------------------------- 代理


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    """把 `/api/*` 透传到网关，保持方法、查询串、请求体与状态码。

    下游不可达映射 502、超时映射 504（与文档 §4.3 的错误约定一致），
    并在 `detail` 里带上目标地址，演示现场一眼能看出是哪儿没起。
    """
    target = resolve_gateway(request)
    url = f"{target}/api/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()
    headers = {}
    if body:
        headers["content-type"] = request.headers.get("content-type", "application/json")

    # 对话链路慢（要过 LLM），用契约里的对话超时；其余走工具超时
    timeout = CHAT_TIMEOUT if path.startswith("chat") else TOOL_TIMEOUT

    client: httpx.AsyncClient = request.app.state.http
    try:
        resp = await client.request(
            request.method,
            url,
            content=body or None,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT),
        )
    except httpx.TimeoutException:
        log.warning("代理超时：%s %s", request.method, url)
        return JSONResponse(
            status_code=504,
            content={"detail": f"网关响应超时（>{timeout:.0f}s）：{url}"},
        )
    except httpx.HTTPError as exc:
        log.warning("网关不可达：%s %s（%s）", request.method, url, exc)
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"网关不可达：{type(exc).__name__}: {exc}。"
                          f"请确认网关已启动、地址正确（当前目标 {target}）"
            },
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------- 静态页


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
