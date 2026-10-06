"""节点配置。

文档 §4.1 要求：节点 IP 用环境变量注入，**不要写死在代码里**。
端口和超时同样允许环境变量覆盖，方便换机器和调试。
配置来源优先级：**显式环境变量 > 仓库根目录 .env > 代码默认值**；
.env 已被 .gitignore 忽略（key 绝不进库），模板见 .env.example。
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录（common/ 的上一层）
ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file() -> None:
    """把仓库根目录的 .env 写进 os.environ（已显式设置的变量不被覆盖）。

    供 key、各节点 IP 这类本地配置落盘：写一次 .env，各窗口/各节点都生效，
    不必每个终端 setx。python-dotenv 未安装时静默跳过（不挡节点启动）。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


load_dotenv_file()

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
    "kb": os.getenv("KB_HOST", "127.0.0.1"),
}

#: 文档 §4.1 约定的端口
PORTS = {
    "gateway": int(os.getenv("GATEWAY_PORT", 8000)),
    "vllm": int(os.getenv("VLLM_PORT", 8001)),
    "vision-fast": int(os.getenv("VISION_FAST_PORT", 8101)),
    "vision-heavy": int(os.getenv("VISION_HEAVY_PORT", 8102)),
    "kb": int(os.getenv("KB_PORT", 8103)),
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
