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

# 本机服务（vLLM、视觉节点）绝不走系统代理：CN 开发机常开代理时，
# openai/httpx 会把 127.0.0.1 的请求交给代理，代理连不上目标就代答 502，
# 让"LLM 不可达"的判断完全失真。进程级 setdefault，一次设置全局生效。
_no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
for _host in ("localhost", "127.0.0.1"):
    if _host not in _no_proxy:
        _no_proxy = f"{_no_proxy},{_host}" if _no_proxy else _host
os.environ["NO_PROXY"] = os.environ["no_proxy"] = _no_proxy


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
    """构造指向 vLLM 的 ChatOpenAI。api_key 对本地 vLLM 是占位符。

    **temperature 千万别设 0**。贪心解码下，只要上下文里出现过一段重复内容，
    模型"最可能的下一个 token"就是继续复读它——自我强化，一旦开始停不下来。
    实测：连着几轮「喵」，它把同一句冷笑话复读了十几遍、每次只改一两个字；
    而且每轮回复都会进历史，等于每轮给自己加码。权重自带的 generation_config
    是 temperature=0.7 / top_p=0.8 / top_k=20，本来就不是给贪心用的。

    默认取模型作者自己的配置 **0.7 / top_p 0.8**，并叠一个 presence_penalty 压制
    逐字复读。这几个数是实测调出来的，别凭感觉降：

        配置                     相邻两轮回复的相似度（连发 6 轮「喵」）
        temperature=0.0          0.24 → 0.32 → 0.44 → 0.86 → 0.96   ← 收敛成复读
        temperature=0.4          0.24 → 0.32 → 0.42 → 0.86 → 0.96   ← 一模一样，白改
        temperature=0.7 + 惩罚    0.20 → 0.36 → 0.28 → 0.31 → 0.27   ← 一直是新内容

    也就是说 **0.4 这种"折中"毫无用处**：复读的主因不是随机性，而是上下文里已经
    堆了几轮相同的回复，续写它就是概率最高的事；只有把温度提到模型本来该用的
    0.7 才压得住。代价是 4B 的工具调用会比贪心时飘一点，所以别再加高。
    """
    kwargs: dict[str, Any] = dict(
        base_url=vllm_base_url() + "/v1",
        api_key="EMPTY",
        model=model_name(),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        top_p=float(os.getenv("LLM_TOP_P", "0.8")),
        #: vLLM 的 OpenAI 接口支持，直接压制逐字复读
        presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", "0.8")),
        timeout=CHAT_TIMEOUT,
        max_retries=0,  # 快速失败上抛，由网关统一映射 502/504
    )
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


async def probe(http: httpx.AsyncClient) -> dict[str, Any]:
    """探测 vLLM 是否存活（GET /v1/models），给 /api/health 用。

    顺带把 **实际生效的窗口大小**带回来：网关靠它算历史字符预算。
    不能只看 VLLM_MAX_MODEL_LEN 环境变量——改了 .env 但没重启 vLLM 时两者会不一致，
    按环境变量推导出的预算会把请求发成超长，照样被 vLLM 400 拒掉（实测踩过）。
    """
    started = time.perf_counter()
    try:
        resp = await http.get(vllm_base_url() + "/v1/models", timeout=3.0)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        ids = [m.get("id") for m in data]
        window = None
        for m in data:
            try:
                window = int(m.get("max_model_len"))
                break
            except (TypeError, ValueError):
                continue
        return {
            "ok": True, "models": ids, "max_model_len": window,
            "elapsed_ms": elapsed, "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - 探活不能抛，必须给聚合响应
        return {
            "ok": False,
            "models": [],
            "max_model_len": None,
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
