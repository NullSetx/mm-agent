"""vLLM 对接：客户端工厂、探活、启动命令。

vLLM 本体只跑在 Linux 环境（16GB 目标机 / WSL2，文档 §8），Windows 侧不装。
本模块负责三件事：

1. `build_chat_model`：给 Agent 造指向 vLLM OpenAI 兼容端点的 ChatOpenAI；
2. `probe`：GET /v1/models 探活，供 /api/health 聚合；
3. `serve_command`：生成 vLLM 启动命令（python -m llm_node.llm 打印）。

WSL2 上必须带 VLLM_WSL2_ENABLE_PIN_MEMORY=1（vLLM 默认在 WSL 上禁用
锁页内存，本机实测可用，见 README「已知问题」）。

换模型只动环境变量：本地开发默认 Qwen2.5-VL-3B-Instruct-AWQ（能自发
发起工具调用的最小多模态模型，2B 及以下不会输出 tool_call），16GB 目标机
把 VLLM_MODEL_PATH 指到 Qwen2-VL-7B-AWQ 即可，代码零改动。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from common.config import CHAT_TIMEOUT, HOSTS, PORTS, ROOT, WEIGHTS_DIR


def vllm_base_url() -> str:
    """vLLM 的 OpenAI 兼容基址。

    不走 common.config.base_url：那个函数对 "vllm" 键会静默错路由到
    vision-heavy（common/config.py 的现成问题，已报群里，修不修三人定）。
    """
    return f"http://{HOSTS['llm']}:{PORTS['vllm']}"


#: 对外模型名。vLLM 用 --served-model-name 固定，与权重路径解耦：
#: 换 7B 时只改启动命令里的路径，这里的名字和 Agent 代码都不用动
DEFAULT_MODEL_NAME = "qwen2.5-vl-3b-awq"


def model_name() -> str:
    return os.getenv("VLLM_MODEL_NAME", DEFAULT_MODEL_NAME)


def build_chat_model(**overrides: Any) -> ChatOpenAI:
    """构造指向 vLLM 的 ChatOpenAI。api_key 对本地 vLLM 是占位符。"""
    kwargs: dict[str, Any] = dict(
        base_url=vllm_base_url() + "/v1",
        api_key="EMPTY",
        model=model_name(),
        temperature=0.0,
        timeout=CHAT_TIMEOUT,
        max_retries=0,  # 快速失败上抛，由网关统一映射 502/504
    )
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


async def probe(http: httpx.AsyncClient) -> dict[str, Any]:
    """探测 vLLM 是否存活（GET /v1/models），给 /api/health 用。"""
    started = time.perf_counter()
    try:
        resp = await http.get(vllm_base_url() + "/v1/models", timeout=3.0)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        resp.raise_for_status()
        ids = [m.get("id") for m in resp.json().get("data", [])]
        return {"ok": True, "models": ids, "elapsed_ms": elapsed, "error": None}
    except Exception as exc:  # noqa: BLE001 - 探活不能抛，必须给聚合响应
        return {
            "ok": False,
            "models": [],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _to_posix_path(p: Path | str) -> str:
    """Windows 盘符路径转 WSL 路径（F:\\x → /mnt/f/x），Linux 路径原样返回。
    打印出来的启动命令在 WSL 里可直接粘贴执行。"""
    s = str(p).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/", s)
    if m:
        return f"/mnt/{m.group(1).lower()}/" + s[3:]
    return s


def serve_command() -> str:
    """生成 vLLM 启动命令（在 WSL2 / Linux 侧执行）。

    Qwen-VL 家族的工具调用走 hermes parser，--enable-auto-tool-choice
    必须带上，否则 Agent 拿不到 tool_calls。
    --chat-template 是必须的：VL 权重自带的模板不支持 tools（模型根本
    看不到工具定义，永远不会自发调用），这里用 Qwen2.5 的 tools 感知模板
    （llm_node/qwen25_tools_chat_template.jinja）替换。
    """
    model_path = os.getenv(
        "VLLM_MODEL_PATH", str(WEIGHTS_DIR / "Qwen2.5-VL-3B-Instruct-AWQ")
    )
    max_len = os.getenv("VLLM_MAX_MODEL_LEN", "4096")
    gpu_util = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
    template = ROOT / "llm_node" / "qwen25_tools_chat_template.jinja"
    return (
        f"VLLM_WSL2_ENABLE_PIN_MEMORY=1 "  # WSL2 必需，Linux 原生可去掉
        f"vllm serve {_to_posix_path(model_path)} \\\n"
        f"  --served-model-name {model_name()} \\\n"
        f"  --chat-template {_to_posix_path(template)} \\\n"
        f"  --port {PORTS['vllm']} \\\n"
        f"  --max-model-len {max_len} \\\n"
        f"  --gpu-memory-utilization {gpu_util} \\\n"
        f"  --enable-auto-tool-choice --tool-call-parser hermes"
    )


if __name__ == "__main__":
    print(serve_command())
