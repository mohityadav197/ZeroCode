"""Additive SQLAlchemy tables; imported by database.init_db before create_all."""
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, Text

from database import Base


class BlackboardRecord(Base):
    __tablename__ = "blackboard_state"
    session_id = Column(String, primary_key=True)
    state = Column(JSON, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AgentLog(Base):
    __tablename__ = "agent_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, nullable=False, index=True)
    run_id = Column(String, nullable=False, index=True)
    agent_name = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    duration_ms = Column(Float)
    input_size = Column(Integer)
    output_size = Column(Integer)
    error = Column(Text)
    traceback = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
