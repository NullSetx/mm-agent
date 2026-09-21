"""会话持久化：SQLite 存储（write-through）。

文档 §4.3 的会话语义在此落地：同一 session_id 的历史与当前图片
跨网关重启保留。课程项目规模下有意从简（YAGNI）：
- 单机单进程（uvicorn 默认单 worker），标准库 sqlite3 足够；
- 不做会话过期清理与删除接口；
- 存储失败只告警不阻断对话（内存态兜底）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

log = logging.getLogger("llm_node.sessions")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    image       TEXT,
    history     TEXT NOT NULL DEFAULT '[]',
    updated_at  REAL
)
"""


@dataclass
class Session:
    """一个会话 = 消息历史 + 当前图片（最近一张为准）。"""

    history: list[BaseMessage] = field(default_factory=list)
    image: str | None = None


class SessionStore:
    """SQLite 会话存储。每次操作短连接，模式表懒创建。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(_SCHEMA)
        return conn

    def get(self, session_id: str) -> Session:
        """取会话；不存在返回空会话（不落库，首次 save 才写）。"""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT image, history FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            log.warning("会话 %s 读取失败，按空会话继续：%s", session_id, exc)
            return Session()
        if row is None:
            return Session()
        image, history_json = row
        try:
            history = messages_from_dict(json.loads(history_json))
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("会话 %s 历史损坏，按空历史继续：%s", session_id, exc)
            history = []
        return Session(history=history, image=image)

    def save(self, session_id: str, session: Session) -> None:
        """write-through 落盘。失败仅告警：对话主流程不因存储问题失败。"""
        try:
            payload = json.dumps(
                messages_to_dict(session.history), ensure_ascii=False
            )
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions "
                    "(session_id, image, history, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, session.image, payload, time.time()),
                )
            finally:
                conn.commit()
                conn.close()
        except (sqlite3.Error, TypeError) as exc:
            log.warning("会话 %s 保存失败：%s", session_id, exc)
