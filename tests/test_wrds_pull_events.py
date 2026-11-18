"""Offline checks of the retry logic in scripts/wrds_pull_events.py (no WRDS connection)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("wrds")

SPEC = importlib.util.spec_from_file_location(
    "wrds_pull_events", Path(__file__).resolve().parents[1] / "scripts" / "wrds_pull_events.py")
pull = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pull)


class FakeConnection:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


class FakeDB:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0
        self.connection = FakeConnection()

    def raw_sql(self, sql):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("SSL SYSCALL error: Operation timed out")
        return pd.DataFrame({"x": [1]})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    from keyperson import wrdsio
    monkeypatch.setattr(wrdsio.time, "sleep", lambda s: None)


def test_fetch_retries_after_rollback():
    db = FakeDB(failures=1)
    df = pull.fetch(db, "SELECT 1", "test")
    assert len(df) == 1 and db.calls == 2 and db.connection.rollbacks == 1


def test_fetch_gives_up_after_two_failures():
    db = FakeDB(failures=5)
    with pytest.raises(RuntimeError):
        pull.fetch(db, "SELECT 1", "test")
    assert db.calls == 2 and db.connection.rollbacks == 2


def test_year_sql_and_keepalives():
    sql = pull.year_sql("ravenpack_dj", "rpa_djpr_equities", 2005)
    assert "ravenpack_dj.rpa_djpr_equities_2005" in sql and "'executive-scandal'" in sql
    assert pull.CONNECT_ARGS["keepalives"] == 1 and pull.CONNECT_ARGS["sslmode"] == "require"
