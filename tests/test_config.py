"""common/config 基础测试：.env 加载与优先级。"""

from __future__ import annotations

import os

from common import config


def test_dotenv_loads_and_env_wins(tmp_path, monkeypatch):
    """.env 里的变量进 os.environ；显式环境变量优先，不被 .env 覆盖。"""
    (tmp_path / ".env").write_text(
        "KB_TEST_FROM_DOTENV=hello\n"
        "KB_EMBEDDING_API_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("KB_EMBEDDING_API_KEY", "from-shell")

    config.load_dotenv_file()

    try:
        assert os.environ["KB_TEST_FROM_DOTENV"] == "hello"      # .env 生效
        assert os.environ["KB_EMBEDDING_API_KEY"] == "from-shell"  # 显式 env 优先
    finally:
        monkeypatch.delenv("KB_TEST_FROM_DOTENV", raising=False)


def test_dotenv_missing_file_is_noop(tmp_path, monkeypatch):
    """没有 .env 不报错（节点照常起）。"""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    config.load_dotenv_file()  # 不应抛异常
