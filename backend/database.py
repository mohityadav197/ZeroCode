# ============================================
# ZeroCode — database.py
# PostgreSQL database connection and
# SQLAlchemy setup with session management.
# ============================================

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://zerocode:zerocode123@localhost:5433/zerocode_db",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_id():
    return str(uuid.uuid4())[:8]


class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    filename = Column(String)
    original_filename = Column(String)
    rows = Column(Integer)
    columns = Column(Integer)
    status = Column(String, default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(String, primary_key=True)
    session_id = Column(String)
    quality_score = Column(Integer)
    problem_type = Column(String)
    target_column = Column(String)
    rows = Column(Integer)
    columns = Column(Integer)
    insights = Column(JSON)
    column_types = Column(JSON)
    missing_values = Column(JSON)
    outliers = Column(JSON)
    target_balance = Column(JSON)
    chart_paths = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class MLResult(Base):
    __tablename__ = "ml_results"
    id = Column(String, primary_key=True)
    session_id = Column(String)
    best_model_name = Column(String)
    best_accuracy = Column(Float)
    best_f1 = Column(Float)
    best_params = Column(JSON)
    leaderboard = Column(JSON)
    preprocessing_log = Column(JSON)
    model_reasoning = Column(JSON)
    tuning_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class SHAPResult(Base):
    __tablename__ = "shap_results"
    id = Column(String, primary_key=True)
    session_id = Column(String)
    status = Column(String)
    feature_importance = Column(JSON)
    explanation = Column(String)
    chart_paths = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class GeneratedFile(Base):
    __tablename__ = "generated_files"
    id = Column(String, primary_key=True)
    session_id = Column(String)
    file_type = Column(String)
    file_path = Column(String)
    is_ready = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True)
    session_id = Column(String)
    role = Column(String)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
