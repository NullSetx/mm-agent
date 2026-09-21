"""会话持久化测试：SQLite round-trip、覆盖更新、缺失会话、损坏历史容错。"""

from langchain_core.messages import AIMessage, HumanMessage

from llm_node.sessions import Session, SessionStore


def test_round_trip(tmp_path):
    """存 → 新实例读（模拟网关重启）→ 消息与图片逐项还原。"""
    store = SessionStore(tmp_path / "sessions.db")
    store.save(
        "s1",
        Session(
            history=[
                HumanMessage(content="图里有什么"),
                AIMessage(content="有一辆公交车。"),
            ],
            image="aGVsbG8=",
        ),
    )

    reopened = SessionStore(tmp_path / "sessions.db")  # 模拟重启
    got = reopened.get("s1")
    assert [m.content for m in got.history] == ["图里有什么", "有一辆公交车。"]
    assert isinstance(got.history[0], HumanMessage)
    assert isinstance(got.history[1], AIMessage)
    assert got.image == "aGVsbG8="


def test_get_missing_returns_empty(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    got = store.get("nope")
    assert got.history == []
    assert got.image is None


def test_save_overwrites(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.save("s1", Session(history=[HumanMessage(content="v1")]))
    store.save("s1", Session(history=[HumanMessage(content="v2")], image="img"))

    got = store.get("s1")
    assert [m.content for m in got.history] == ["v2"]
    assert got.image == "img"


def test_corrupt_history_degrades_to_empty(tmp_path):
    """历史 JSON 损坏时按空历史继续，不炸接口。"""
    store = SessionStore(tmp_path / "sessions.db")
    store.save("s1", Session(history=[HumanMessage(content="ok")]))
    # 手写坏数据模拟损坏
    import sqlite3

    conn = sqlite3.connect(tmp_path / "sessions.db")
    conn.execute("UPDATE sessions SET history = '{not json' WHERE session_id = 's1'")
    conn.commit()
    conn.close()

    got = store.get("s1")
    assert got.history == []
    assert got.image is None
