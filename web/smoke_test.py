"""演示前端（web/）端到端冒烟测试：一条命令起 mock 三件套并走代理验证全链路。

跑法（仓库根目录，用项目 conda 环境的 python）：

    python web/smoke_test.py

它会依次启动：vision_fast(:8101, NODE_MOCK=1) → 网关(:8000, NODE_MOCK=1) → 本前端(:8103)，
然后**只经过前端代理**打一遍接口，最后把服务全部关掉。不需要模型权重、不需要 vLLM。

默认使用契约端口；端口被占时用环境变量改（脚本会自动透传给三个服务）：
    WEB_PORT=18103 GATEWAY_PORT=18000 VISION_FAST_PORT=18101 python web/smoke_test.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
FAST_PORT = int(os.getenv("VISION_FAST_PORT", "8101"))
WEB_PORT = int(os.getenv("WEB_PORT", "8103"))
WEB = f"http://127.0.0.1:{WEB_PORT}"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


def http(method: str, url: str, body=None, headers=None, timeout=90):
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def as_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


def wait_http(url: str, tries: int = 200, sleep: float = 0.5) -> bool:
    """等某个服务就绪。默认给 100 秒——首次启动要导入 langchain/fastapi，慢机器上要几十秒。"""
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(sleep)
    return False


def dump_log(proc: subprocess.Popen, label: str) -> None:
    """把子进程输出读出来打印（进程还活着也照样读）。"""
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        out = proc.communicate(timeout=15)[0]
    except Exception:
        proc.kill()
        out = "(无法读取输出)"
    print(f"--- {label} 输出 ---\n{(out or '')[:2500]}", flush=True)


def make_image_b64() -> str:
    """用 cv2 造一张小图（conda 环境里有 opencv；连它也没装就退回一张 1×1 PNG）。"""
    code = (
        "import base64, cv2, numpy as np;"
        "img = np.zeros((240, 320, 3), np.uint8);"
        "img[60:180, 40:280] = (200, 190, 180);"
        "ok, buf = cv2.imencode('.jpg', img);"
        "print(base64.b64encode(buf.tobytes()).decode())"
    )
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    one_px = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAF"
              "BQMBmUG0wwAAAABJRU5ErkJggg==")
    return one_px


def start(module: str, port: int, extra_env: dict | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NODE_MOCK"] = "1"
    env.update(extra_env or {})
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    return proc


def main() -> int:
    print("=== 启动 mock 三件套（无权重、无 vLLM），按顺序等就绪 ===", flush=True)
    procs: list[subprocess.Popen] = []
    plan = [
        ("vision_fast", "vision_fast.server:app", FAST_PORT, f"http://127.0.0.1:{FAST_PORT}/health"),
        ("gateway", "llm_node.gateway:app", GATEWAY_PORT, f"http://127.0.0.1:{GATEWAY_PORT}/api/health"),
        ("web", "web.server:app", WEB_PORT, f"{WEB}/healthz"),
    ]

    try:
        for label, module, port, url in plan:
            proc = start(module, port)
            procs.append(proc)
            ok = wait_http(url)
            check(f"{label} 启动并就绪", ok, url)
            if not ok:
                dump_log(proc, label)
                return 1

        print("\n=== 静态页与配置 ===", flush=True)
        status, raw = http("GET", f"{WEB}/")
        check("GET / 返回页面", status == 200 and b"mm-agent" in raw, f"HTTP {status}")
        status, raw = http("GET", f"{WEB}/static/app.js")
        check("GET /static/app.js", status == 200 and len(raw) > 1000, f"HTTP {status}, {len(raw)} B")
        status, raw = http("GET", f"{WEB}/static/style.css")
        check("GET /static/style.css", status == 200 and len(raw) > 500, f"HTTP {status}, {len(raw)} B")
        status, raw = http("GET", f"{WEB}/__config")
        cfg = as_json(raw) or {}
        check("GET /__config", status == 200 and "gateway" in cfg, f"gateway={cfg.get('gateway')}")

        print("\n=== 经代理调网关（同源链路）===", flush=True)
        status, raw = http("GET", f"{WEB}/api/health")
        health = as_json(raw) or {}
        nodes = health.get("nodes", {})
        fast = nodes.get("vision-fast", {})
        check("GET /api/health 透传", status == 200 and fast.get("ok") is True,
              f"vision-fast ok={fast.get('ok')} tools={fast.get('tools')}")
        check("health 里 tools 被聚合", "detect" in (health.get("gateway", {}).get("tools") or []),
              str(health.get("gateway", {}).get("tools")))

        status, raw = http("GET", f"{WEB}/api/tools")
        tools = as_json(raw) or {}
        names = [t.get("name") for t in tools.get("tools", [])]
        check("GET /api/tools 透传", status == 200 and "detect" in names and "classify" in names, str(names))

        status, raw = http("POST", f"{WEB}/api/tools/refresh", body={})
        rep = as_json(raw) or {}
        rep_nodes = rep.get("nodes", {})
        fast_rep = rep_nodes.get("vision-fast", {})
        # 注意：网关返回的 ok 是「所有节点都成功」的聚合值；本测试没起 vision-heavy，
        # 所以聚合必为 False——这里只校验 vision-fast 确实被发现（这才是本节点关心的事）。
        node_brief = {k: v.get("ok") for k, v in rep_nodes.items()}
        check("POST /api/tools/refresh（vision-fast 被发现；vision-heavy 未启动属预期）",
              status == 200 and fast_rep.get("ok") is True
              and {"detect", "classify"}.issubset(set(fast_rep.get("tools") or [])),
              f"aggregate_ok={rep.get('ok')}, nodes={node_brief}")

        img = make_image_b64()
        status, raw = http("POST", f"{WEB}/api/invoke",
                           body={"tool": "detect", "image": img, "params": {"conf": "0.3"}})
        body = as_json(raw) or {}
        res = body.get("result") or {}
        check("POST /api/invoke detect（含字符串参数强转）", status == 200 and body.get("ok") is True,
              f"HTTP {status}, mock={res.get('mock')}, conf={res.get('conf_threshold')}")
        check("detect 返回 2 个框（mock 结构）", len(res.get("boxes") or []) == 2,
              f"count={res.get('count')}")

        status, raw = http("POST", f"{WEB}/api/invoke", body={"tool": "classify", "image": img, "params": {"topk": 3}})
        body = as_json(raw) or {}
        check("POST /api/invoke classify", status == 200 and body.get("ok") is True,
              str((body.get("result") or {}).get("top1")))

        status, raw = http("POST", f"{WEB}/api/chat",
                           body={"session_id": "smoke-1", "message": "图里有什么？", "image": img})
        chat = as_json(raw) or {}
        check("POST /api/chat（mock 回复 + 工具调用记录）", status == 200 and bool(chat.get("reply")),
              f"reply={str(chat.get('reply'))[:40]!r}, tool_calls={[c.get('tool') for c in chat.get('tool_calls', [])]}")

        status, raw = http("POST", f"{WEB}/api/invoke", body={"tool": "不存在的工具"})
        check("未知工具 → 404（错误码按契约透传）", status == 404, f"HTTP {status}")

        print("\n=== 动态代理目标 ===", flush=True)
        status, raw = http("GET", f"{WEB}/api/health", headers={"X-Gateway-Target": "127.0.0.1:9"})
        detail = (as_json(raw) or {}).get("detail", "")
        check("请求头指定的错误目标 → 502（证明目标按请求动态解析）",
              status == 502 and "网关不可达" in detail, f"HTTP {status}")
        status, raw = http("GET", f"{WEB}/api/health")
        check("去掉请求头后自动回到默认目标", status == 200, f"HTTP {status}")

        print("\n=== 结果 ===", flush=True)
        failed = [n for n, ok, _ in results if not ok]
        print(f"  共 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}", flush=True)
        if failed:
            print("  失败项：" + "; ".join(failed), flush=True)
            return 1
        print("  全部通过", flush=True)
        return 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except Exception:
                p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
