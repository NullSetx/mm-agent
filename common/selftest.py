"""common 契约自检。

不依赖任何模型权重，也不依赖队友的节点——用 TestClient 起一个带 mock 工具的
节点，逐条断言 `docs/分工与接口约定.md` §4 约定的行为。

跑法：

    python -m common.selftest

任何时候 `common/` 被改动，或者 B、A 怀疑自己的节点不符合契约，跑一遍这个即可。
"""

from __future__ import annotations

import base64
import logging
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

from common.node import build_app, decode_image
from common.registry import ToolError, ToolRegistry

# 自检里会故意触发工具异常，节点会打完整 traceback。那是生产环境想要的行为，
# 但在这里会盖住测试结果，所以临时压掉。
logging.getLogger("common.node").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------- 测试用工具


def make_registry() -> ToolRegistry:
    """构造一个带两个 mock 工具的隔离注册表。"""
    reg = ToolRegistry(node="test-node")

    @reg.register(
        name="echo",
        description="回显：把图片尺寸和收到的参数原样返回，用于验证参数处理",
        needs_image=True,
        params={"conf": 0.25, "tag": "x", "flag": False, "classes": None},
    )
    def echo(image, conf=0.25, tag="x", flag=False, classes=None):
        return {
            "shape": list(image.shape),
            "conf": conf,
            "conf_type": type(conf).__name__,
            "tag": tag,
            "flag": flag,
            "classes": classes,
        }

    @reg.register(
        name="no_image",
        description="不需要图片的工具",
        needs_image=False,
        params={"n": 1},
    )
    def no_image(n=1):
        return {"n": n, "n_type": type(n).__name__}

    @reg.register(name="boom", description="总是抛异常，用于验证错误包装", needs_image=False)
    def boom():
        raise RuntimeError("故意炸的")

    return reg


def sample_image_b64(with_prefix: bool = False, pad: bool = True) -> str:
    img = np.zeros((20, 30, 3), np.uint8)
    img[:, :] = (50, 100, 150)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    text = base64.b64encode(buf.tobytes()).decode()
    if not pad:
        text = text.rstrip("=")
    if with_prefix:
        return f"data:image/jpeg;base64,{text}"
    return text


# ---------------------------------------------------------------- 断言框架

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, fn) -> None:
    try:
        fn()
    except AssertionError as exc:
        RESULTS.append((label, False, str(exc) or "断言失败"))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((label, False, f"{type(exc).__name__}: {exc}"))
    else:
        RESULTS.append((label, True, ""))


# ---------------------------------------------------------------- 各项测试


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["node"] == "test-node", body
    assert body["status"] == "ok", body
    assert set(body["tools"]) == {"echo", "no_image", "boom"}, body
    assert isinstance(body["detail"], dict), body


def test_tools(client: TestClient) -> None:
    r = client.get("/tools")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["node"] == "test-node", body
    specs = {t["name"]: t for t in body["tools"]}
    assert set(specs) == {"echo", "no_image", "boom"}, specs
    echo = specs["echo"]
    assert echo["needs_image"] is True, echo
    assert echo["params"]["conf"] == 0.25, echo
    assert echo["params"]["classes"] is None, echo
    assert specs["no_image"]["needs_image"] is False, specs["no_image"]


def test_invoke_ok(client: TestClient) -> None:
    r = client.post("/invoke", json={"tool": "echo", "image": sample_image_b64()})
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["ok"] is True, body
    assert body["error"] is None, body
    assert body["result"]["shape"] == [20, 30, 3], body
    assert isinstance(body["elapsed_ms"], (int, float)), body


def test_invoke_data_uri_prefix(client: TestClient) -> None:
    """带 `data:image/...;base64,` 前缀要能解析（文档 §4.1 明确允许）。"""
    r = client.post(
        "/invoke", json={"tool": "echo", "image": sample_image_b64(with_prefix=True)}
    )
    body = r.json()
    assert body["ok"] is True, body
    assert body["result"]["shape"] == [20, 30, 3], body


def test_invoke_missing_padding(client: TestClient) -> None:
    """base64 缺 padding 也要能解析——实际调用里很常见。"""
    r = client.post(
        "/invoke", json={"tool": "echo", "image": sample_image_b64(pad=False)}
    )
    body = r.json()
    assert body["ok"] is True, body


def test_param_coercion(client: TestClient) -> None:
    """LLM 常把数字发成字符串，要按默认值类型强转。"""
    r = client.post(
        "/invoke",
        json={
            "tool": "echo",
            "image": sample_image_b64(),
            "params": {"conf": "0.5", "flag": "true", "tag": 123},
        },
    )
    body = r.json()
    assert body["ok"] is True, body
    res = body["result"]
    assert res["conf"] == 0.5 and res["conf_type"] == "float", res
    # bool 必须判断在 int 之前，否则 "true" 会被当成数字
    assert res["flag"] is True, res
    assert res["tag"] == "123", res


def test_param_filtering(client: TestClient) -> None:
    """LLM 臆造的参数要被丢弃，而不是抛 TypeError。"""
    r = client.post(
        "/invoke",
        json={
            "tool": "echo",
            "image": sample_image_b64(),
            "params": {"conf": 0.9, "不存在的参数": 1, "another": "x"},
        },
    )
    body = r.json()
    assert body["ok"] is True, body
    assert body["result"]["conf"] == 0.9, body


def test_param_none_default(client: TestClient) -> None:
    """默认值为 None 的参数无法推断类型，原样透传。"""
    r = client.post(
        "/invoke",
        json={"tool": "echo", "image": sample_image_b64(), "params": {"classes": [1, 2]}},
    )
    body = r.json()
    assert body["ok"] is True, body
    assert body["result"]["classes"] == [1, 2], body


def test_no_image_tool(client: TestClient) -> None:
    """needs_image=False 的工具不传图片也能调。"""
    r = client.post("/invoke", json={"tool": "no_image", "params": {"n": "3"}})
    body = r.json()
    assert body["ok"] is True, body
    assert body["result"]["n"] == 3 and body["result"]["n_type"] == "int", body


def test_missing_image(client: TestClient) -> None:
    """需要图片却没传 → ok=false，但 HTTP 仍是 200。"""
    r = client.post("/invoke", json={"tool": "echo"})
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["ok"] is False, body
    assert body["result"] is None, body
    assert "image" in (body["error"] or ""), body


def test_invalid_base64(client: TestClient) -> None:
    r = client.post("/invoke", json={"tool": "echo", "image": "!!!不是base64!!!"})
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["ok"] is False, body
    assert "base64" in (body["error"] or ""), body


def test_base64_not_an_image(client: TestClient) -> None:
    """合法 base64 但内容不是图片。"""
    not_image = base64.b64encode(b"hello world, not an image").decode()
    r = client.post("/invoke", json={"tool": "echo", "image": not_image})
    body = r.json()
    assert body["ok"] is False, body
    assert "图片" in (body["error"] or ""), body


def test_tool_exception_wrapped(client: TestClient) -> None:
    """工具内部异常必须被包装成 ok=false，且 HTTP 仍为 200（不拖垮节点）。"""
    r = client.post("/invoke", json={"tool": "boom"})
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["ok"] is False, body
    assert body["result"] is None, body
    assert "RuntimeError" in (body["error"] or ""), body
    assert "故意炸的" in (body["error"] or ""), body


def test_unknown_tool_404(client: TestClient) -> None:
    """未知工具是路由问题，按文档 §4.3 返回 404。"""
    r = client.post("/invoke", json={"tool": "不存在的工具"})
    assert r.status_code == 404, f"期望 404，实际 {r.status_code}"


def test_duplicate_tool_name_rejected() -> None:
    reg = make_registry()
    try:
        reg.register(name="echo", description="重名")(lambda: None)
    except ToolError:
        return
    raise AssertionError("重名工具应当被拒绝")


def test_bad_param_name_rejected() -> None:
    """声明的参数不在函数签名里 → 登记时就该报错，而不是等调用时静默丢弃。"""
    reg = ToolRegistry(node="t")
    try:
        reg.register(name="bad", description="x", needs_image=False,
                     params={"conf": 0.5})(lambda: None)
    except ToolError:
        return
    raise AssertionError("签名里没有的参数名应当被拒绝")


def test_missing_image_param_rejected() -> None:
    """声明 needs_image=True 但函数没有 image 形参 → 登记时报错。"""
    reg = ToolRegistry(node="t")
    try:
        reg.register(name="bad2", description="x", needs_image=True)(lambda foo: None)
    except ToolError:
        return
    raise AssertionError("缺 image 形参应当被拒绝")


def test_decode_image_direct() -> None:
    """直接测解码函数，确认返回的是 cv2 约定的 BGR 数组。"""
    img = decode_image(sample_image_b64(with_prefix=True, pad=False))
    assert isinstance(img, np.ndarray), type(img)
    assert img.shape == (20, 30, 3), img.shape
    assert img.dtype == np.uint8, img.dtype


# ---------------------------------------------------------------- 入口


TESTS = [
    ("GET /health 结构与字段", test_health),
    ("GET /tools 自报工具清单", test_tools),
    ("POST /invoke 成功路径", test_invoke_ok),
    ("base64 带 data: 前缀", test_invoke_data_uri_prefix),
    ("base64 缺 padding", test_invoke_missing_padding),
    ("参数类型强转（字符串→数字/bool）", test_param_coercion),
    ("参数过滤（丢弃臆造参数）", test_param_filtering),
    ("默认值为 None 的参数透传", test_param_none_default),
    ("needs_image=False 不传图片", test_no_image_tool),
    ("缺图片 → ok=false 且 HTTP 200", test_missing_image),
    ("非法 base64 → ok=false", test_invalid_base64),
    ("合法 base64 但非图片 → ok=false", test_base64_not_an_image),
    ("工具异常 → ok=false 且 HTTP 200", test_tool_exception_wrapped),
    ("未知工具 → HTTP 404", test_unknown_tool_404),
    ("重名工具被拒绝", test_duplicate_tool_name_rejected),
    ("签名中没有的参数名被拒绝", test_bad_param_name_rejected),
    ("缺 image 形参被拒绝", test_missing_image_param_rejected),
    ("decode_image 返回 BGR 数组", test_decode_image_direct),
]


def main() -> int:
    app = build_app("test-node", registry=make_registry())
    client = TestClient(app)
    for label, fn in TESTS:
        if fn in (test_duplicate_tool_name_rejected, test_bad_param_name_rejected,
                  test_missing_image_param_rejected, test_decode_image_direct):
            check(label, fn)
        else:
            check(label, lambda f=fn: f(client))

    width = max(len(label) for label, _ in TESTS)
    passed = 0
    for label, ok, msg in RESULTS:
        mark = "✅" if ok else "❌"
        line = f"  {mark} {label.ljust(width)}"
        if not ok:
            line += f"  ← {msg}"
        print(line)
        passed += ok

    total = len(RESULTS)
    print()
    if passed == total:
        print(f"契约自检全部通过（{passed}/{total}）")
        return 0
    print(f"契约自检失败：{total - passed}/{total} 项未通过")
    return 1


if __name__ == "__main__":
    sys.exit(main())
