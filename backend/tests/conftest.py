import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATABASE_URL"] = "sqlite:///./test_zerocode.db"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"


import pytest


@pytest.fixture(autouse=True)
def database_tables():
    from database import Base, engine, init_db
    init_db()
    yield
    Base.metadata.drop_all(engine)
