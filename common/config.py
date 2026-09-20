"""节点配置。

文档 §4.1 要求：节点 IP 用环境变量注入，**不要写死在代码里**。
端口和超时同样允许环境变量覆盖，方便换机器和调试。
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录（common/ 的上一层）
ROOT = Path(__file__).resolve().parent.parent

# 模型权重与数据。已被 .gitignore 覆盖，各人本地准备
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- 网络

#: 各节点主机。默认本机，跨机联调时注入真实 IP，例如：
#:   VISION_FAST_HOST=192.168.1.101 VISION_HEAVY_HOST=192.168.1.102 uvicorn ...
HOSTS = {
    "llm": os.getenv("LLM_HOST", "127.0.0.1"),
    "vision-fast": os.getenv("VISION_FAST_HOST", "127.0.0.1"),
    "vision-heavy": os.getenv("VISION_HEAVY_HOST", "127.0.0.1"),
}

#: 文档 §4.1 约定的端口
PORTS = {
    "gateway": int(os.getenv("GATEWAY_PORT", 8000)),
    "vllm": int(os.getenv("VLLM_PORT", 8001)),
    "vision-fast": int(os.getenv("VISION_FAST_PORT", 8101)),
    "vision-heavy": int(os.getenv("VISION_HEAVY_PORT", 8102)),
}


def base_url(node: str) -> str:
    """拼出某个节点的基址，如 http://192.168.1.102:8102"""
    key = node if node in HOSTS else "vision-heavy"
    return f"http://{HOSTS[key]}:{PORTS[key]}"


# ---------------------------------------------------------------- 超时

#: 单次工具调用超时（文档 §4.1）
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT", 30))
#: 对话超时（文档 §4.1）
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", 60))


# ---------------------------------------------------------------- 运行模式

def mock_enabled() -> bool:
    """是否处于 mock 模式（文档 §6.2「先跑通 mock，再上模型」）。

    置位后各工具返回结构合法的占位结果，让三方在没有模型权重时也能联调全链路。
    """
    return os.getenv("NODE_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}
