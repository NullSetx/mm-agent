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


def test_exists_and_list_sessions(tmp_path):
    """exists 区分 404；list_sessions 新→旧，带轮数与首条提问摘要。"""
    import time

    store = SessionStore(tmp_path / "sessions.db")
    assert not store.exists("s1")
    store.save("s1", Session(history=[
        HumanMessage(content="图里有什么？"),
        AIMessage(content="", tool_calls=[
            {"name": "detect", "args": {"conf": 0.3}, "id": "c1", "type": "tool_call"},
        ]),
        AIMessage(content="有一辆公交车。"),
    ]))
    time.sleep(0.01)  # updated_at 秒级精度，保证排序稳定
    store.save("s2", Session(history=[HumanMessage(content="第二段对话")]))

    assert store.exists("s1")
    sessions = store.list_sessions()
    assert [s["session_id"] for s in sessions] == ["s2", "s1"]  # 新→旧
    s1 = next(s for s in sessions if s["session_id"] == "s1")
    assert s1["turns"] == 1 and s1["preview"] == "图里有什么？"
    assert s1["updated_at"] > 0


def test_delete_session(tmp_path):
    """删除清掉记录；不存在返回 False；删后再 get 得到空会话。"""
    store = SessionStore(tmp_path / "sessions.db")
    store.save("s1", Session(history=[HumanMessage(content="x")], image="img"))
    assert store.delete("s1") is True
    assert store.exists("s1") is False
    assert store.delete("s1") is False
    assert store.get("s1").history == []


def test_list_sessions_corrupt_row_survives(tmp_path):
    """列表里混进一条损坏历史：该会话 preview 为空，但其余照常返回。"""
    import sqlite3

    store = SessionStore(tmp_path / "sessions.db")
    store.save("good", Session(history=[HumanMessage(content="好的会话")]))
    store.save("bad", Session(history=[HumanMessage(content="坏的会话")]))
    conn = sqlite3.connect(tmp_path / "sessions.db")
    conn.execute("UPDATE sessions SET history = '{bad' WHERE session_id = 'bad'")
    conn.commit()
    conn.close()

    sessions = {s["session_id"]: s for s in store.list_sessions()}
    assert sessions["good"]["preview"] == "好的会话"
    assert sessions["bad"]["preview"] == ""
