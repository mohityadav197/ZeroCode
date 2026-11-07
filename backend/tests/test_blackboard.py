from concurrent.futures import ThreadPoolExecutor

import pytest

from blackboard import Blackboard


def test_concurrent_updates_persist_and_sessions_are_isolated():
    board = Blackboard("first")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: board.add_tokens(1), range(20)))
    assert Blackboard("first").get_full_state()["tokens_used"] == 20
    assert Blackboard("second").get_full_state()["tokens_used"] == 0
    with pytest.raises(ValueError):
        board.update("unknown", {})


def test_get_fields_returns_only_the_requested_sections():
    board = Blackboard("fields")
    board.add_tokens(5)
    assert board.get_fields("tokens_used", "ml_result") == {"tokens_used": 5, "ml_result": None}
