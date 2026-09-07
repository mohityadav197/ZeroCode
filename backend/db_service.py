# ============================================
# ZeroCode — db_service.py
# Database service layer — all save and
# retrieve operations for every agent result.
# ============================================

from datetime import datetime

from database import AnalysisResult, ChatMessage, GeneratedFile, MLResult, SHAPResult, Session, generate_id


def create_session(db, filename, rows, columns):
    session = Session(
        session_id=generate_id(),
        filename=filename,
        original_filename=filename,
        rows=rows,
        columns=columns,
        status="uploaded",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def update_session_status(db, session_id, status):
    session = db.query(Session).filter(Session.session_id == session_id).first()
    if session is None:
        return None
    session.status = status
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


def save_analysis_result(db, session_id, analysis_dict):
    profile = analysis_dict.get("profile", {})
    problem_type = analysis_dict.get("problem_type") or {}
    target_suggestion = analysis_dict.get("target_suggestion") or {}
    result = AnalysisResult(
        id=generate_id(),
        session_id=session_id,
        quality_score=profile.get("quality_score"),
        problem_type=problem_type.get("problem_type"),
        target_column=target_suggestion.get("suggested_target"),
        rows=profile.get("rows"),
        columns=profile.get("columns"),
        insights=analysis_dict.get("insights"),
        column_types=profile.get("column_types"),
        missing_values=profile.get("missing"),
        outliers=analysis_dict.get("outliers"),
        target_balance=analysis_dict.get("target_balance"),
        chart_paths=analysis_dict.get("charts"),
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_analysis_result(db, session_id):
    return (
        db.query(AnalysisResult)
        .filter(AnalysisResult.session_id == session_id)
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )


def save_ml_result(db, session_id, ml_dict):
    leaderboard = ml_dict.get("leaderboard") or []
    best_entry = next((r for r in leaderboard if r.get("is_best")), leaderboard[0] if leaderboard else {})
    result = MLResult(
        id=generate_id(),
        session_id=session_id,
        best_model_name=ml_dict.get("best_model_name"),
        best_accuracy=best_entry.get("accuracy"),
        best_f1=best_entry.get("f1"),
        best_params=ml_dict.get("best_params"),
        leaderboard=leaderboard,
        preprocessing_log=ml_dict.get("preprocessing_log"),
        model_reasoning=ml_dict.get("model_reasoning"),
        tuning_score=ml_dict.get("tuning_score"),
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_ml_result(db, session_id):
    return (
        db.query(MLResult)
        .filter(MLResult.session_id == session_id)
        .order_by(MLResult.created_at.desc())
        .first()
    )


def save_shap_result(db, session_id, shap_dict):
    result = SHAPResult(
        id=generate_id(),
        session_id=session_id,
        status=shap_dict.get("status"),
        feature_importance=shap_dict.get("feature_importance"),
        explanation=shap_dict.get("explanation"),
        chart_paths=shap_dict.get("charts"),
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_shap_result(db, session_id):
    return (
        db.query(SHAPResult)
        .filter(SHAPResult.session_id == session_id)
        .order_by(SHAPResult.created_at.desc())
        .first()
    )


def save_file_record(db, session_id, file_type, file_path):
    record = GeneratedFile(
        id=generate_id(),
        session_id=session_id,
        file_type=file_type,
        file_path=file_path,
        is_ready=True,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_session_files(db, session_id):
    return (
        db.query(GeneratedFile)
        .filter(GeneratedFile.session_id == session_id, GeneratedFile.is_ready.is_(True))
        .all()
    )


def save_chat_message(db, session_id, role, message):
    entry = ChatMessage(
        id=generate_id(),
        session_id=session_id,
        role=role,
        message=message,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_chat_history(db, session_id, limit=20):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(messages))


def get_all_sessions(db, limit=10):
    return db.query(Session).order_by(Session.created_at.desc()).limit(limit).all()
