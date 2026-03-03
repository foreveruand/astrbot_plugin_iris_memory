"""test_expectation_store.py - ExpectationStore 存储测试"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from iris_memory.proactive.models import FollowUpExpectation
from iris_memory.proactive.storage.expectation_store import ExpectationStore


def _make_exp(group_id: str = "g1", **kwargs) -> FollowUpExpectation:
    defaults = dict(
        session_key=f"u1:{group_id}",
        group_id=group_id,
        trigger_user_id="u1",
        trigger_message="hello",
        bot_reply_summary="hi there",
        followup_window_end=datetime.now() + timedelta(minutes=2),
    )
    defaults.update(kwargs)
    return FollowUpExpectation(**defaults)


@pytest.fixture
def store() -> ExpectationStore:
    return ExpectationStore()


class TestPut:
    """存储期待测试"""

    def test_put_new(self, store: ExpectationStore) -> None:
        exp = _make_exp()
        store.put(exp)
        assert store.count == 1

    def test_put_replaces_same_group(self, store: ExpectationStore) -> None:
        exp1 = _make_exp(group_id="g1")
        exp2 = _make_exp(group_id="g1")
        store.put(exp1)
        store.put(exp2)
        assert store.count == 1
        retrieved = store.get("g1")
        assert retrieved.expectation_id == exp2.expectation_id

    def test_put_different_groups(self, store: ExpectationStore) -> None:
        store.put(_make_exp(group_id="g1"))
        store.put(_make_exp(group_id="g2"))
        assert store.count == 2


class TestGet:
    """获取期待测试"""

    def test_get_existing(self, store: ExpectationStore) -> None:
        exp = _make_exp()
        store.put(exp)
        result = store.get("g1")
        assert result is not None
        assert result.expectation_id == exp.expectation_id

    def test_get_nonexistent(self, store: ExpectationStore) -> None:
        assert store.get("nope") is None

    def test_get_expired_returns_none_and_removes(self, store: ExpectationStore) -> None:
        exp = _make_exp(
            followup_window_end=datetime.now() - timedelta(seconds=1)
        )
        store.put(exp)
        assert store.get("g1") is None
        assert store.count == 0


class TestRemove:
    """移除期待测试"""

    def test_remove_existing(self, store: ExpectationStore) -> None:
        store.put(_make_exp())
        removed = store.remove("g1")
        assert removed is not None
        assert store.count == 0

    def test_remove_nonexistent(self, store: ExpectationStore) -> None:
        removed = store.remove("nope")
        assert removed is None


class TestHasActive:
    """活跃期待检查测试"""

    def test_has_active_true(self, store: ExpectationStore) -> None:
        store.put(_make_exp())
        assert store.has_active("g1") is True

    def test_has_active_false(self, store: ExpectationStore) -> None:
        assert store.has_active("g1") is False

    def test_has_active_expired(self, store: ExpectationStore) -> None:
        store.put(_make_exp(
            followup_window_end=datetime.now() - timedelta(seconds=1)
        ))
        # has_active calls get() which cleans expired
        assert store.has_active("g1") is False


class TestGetAll:
    """获取所有期待测试"""

    def test_get_all_empty(self, store: ExpectationStore) -> None:
        assert store.get_all() == []

    def test_get_all_filters_expired(self, store: ExpectationStore) -> None:
        store.put(_make_exp(group_id="g1"))
        store.put(_make_exp(
            group_id="g2",
            followup_window_end=datetime.now() - timedelta(seconds=1),
        ))
        all_exps = store.get_all()
        assert len(all_exps) == 1
        assert all_exps[0].group_id == "g1"


class TestClear:
    """清除所有期待测试"""

    def test_clear(self, store: ExpectationStore) -> None:
        store.put(_make_exp(group_id="g1"))
        store.put(_make_exp(group_id="g2"))
        cleared = store.clear()
        assert cleared == 2
        assert store.count == 0

    def test_clear_empty(self, store: ExpectationStore) -> None:
        assert store.clear() == 0
