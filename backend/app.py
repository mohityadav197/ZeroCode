# ============================================
# ZeroCode — app.py
# Main FastAPI application — handles all API
# routes, file uploads, and CORS setup.
# ============================================

import os
from typing import Any, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from agents.analysis_agent import AnalysisAgent
from agents.ml_agent import MLAgent
from agents.orchestrator import Orchestrator
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, PROJECT_NAME, UPLOAD_FOLDER
from database import get_db, init_db
from db_service import (
    create_session,
    get_all_sessions,
    get_analysis_result,
    get_chat_history,
    get_ml_result,
    get_session_files,
    get_shap_result,
    save_analysis_result,
    save_chat_message,
    save_file_record,
    save_ml_result,
    save_shap_result,
    update_session_status,
)
from utils.report_generator import ReportGenerator

app = FastAPI(title="ZeroCode API")
analysis_agent = AnalysisAgent()
ml_agent = MLAgent()
report_generator = ReportGenerator()
orchestrator = Orchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@app.on_event("startup")
def startup():
    init_db()


def clean_for_json(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    return obj


@app.get("/health")
def health_check():
    return {"status": "ok", "project": PROJECT_NAME}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: DBSession = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only {ALLOWED_EXTENSIONS} files are allowed.",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    session = create_session(db, file.filename, len(df), len(df.columns))
    orchestrator.update_context("session_id", session.session_id)

    return clean_for_json({
        "success": True,
        "session_id": session.session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    })


@app.get("/files/{filename:path}")
def get_file(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return FileResponse(filepath)


class AnalyzeRequest(BaseModel):
    filename: str
    confirmed_target: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/analyze")
def analyze_file(payload: AnalyzeRequest, db: DBSession = Depends(get_db)):
    filepath = os.path.join(UPLOAD_FOLDER, payload.filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    result = analysis_agent.run_full_analysis(df, confirmed_target=payload.confirmed_target)

    problem_type_str = (result.get("problem_type", {}) or {}).get("problem_type")
    profile = result.get("profile", {}) or {}
    model_recs = ml_agent.generate_model_recommendations(df, problem_type_str, profile, result)
    result["model_recommendations"] = model_recs

    orchestrator.update_context("filename", payload.filename)
    orchestrator.update_context("analysis_result", result)
    orchestrator.update_context("insights", result.get("insights", []))
    orchestrator.update_context(
        "charts_generated",
        list((result.get("charts", {}) or {}).get("charts", {}).keys()),
    )
    orchestrator.update_context("outliers", (result.get("outliers", {}) or {}).get("outliers", {}))
    orchestrator.update_context("target_balance", result.get("target_balance", {}))
    orchestrator.update_context("missing_values", (result.get("profile", {}) or {}).get("missing", {}))
    orchestrator.update_context("quality_score", (result.get("profile", {}) or {}).get("quality_score", 0))
    orchestrator.update_context("model_recommendations", model_recs)
    orchestrator.update_context("domain_info", result.get("domain_info"))

    if payload.session_id:
        save_analysis_result(db, payload.session_id, result)
        update_session_status(db, payload.session_id, "analyzed")
        chart_paths = (result.get("charts") or {}).get("charts") or {}
        for chart_type, chart_path in chart_paths.items():
            save_file_record(db, payload.session_id, f"chart_{chart_type}", chart_path)

    return clean_for_json(result)


class PreprocessingRecommendationsRequest(BaseModel):
    filename: str
    target_col: str
    selected_models: Optional[List[str]] = None


@app.post("/preprocessing-recommendations")
def preprocessing_recommendations(payload: PreprocessingRecommendationsRequest):
    filepath = os.path.join(UPLOAD_FOLDER, payload.filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    if payload.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{payload.target_col}' not found in dataset.")

    profile = analysis_agent.profile_data(df)
    domain_info = analysis_agent.detect_domain(df, profile)
    result = analysis_agent.generate_preprocessing_recommendations(
        df, profile, domain_info,
        target_col=payload.target_col,
        selected_models=payload.selected_models,
    )
    return clean_for_json(result)


class TrainRequest(BaseModel):
    filename: str
    target_col: str
    problem_type: str
    extra_models: Optional[List[str]] = None
    selected_metrics: Optional[List[str]] = None
    user_preprocessing_choices: Optional[dict] = None
    session_id: Optional[str] = None


@app.post("/train")
def train_file(payload: TrainRequest, db: DBSession = Depends(get_db)):
    filepath = os.path.join(UPLOAD_FOLDER, payload.filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    if payload.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{payload.target_col}' not found in dataset.")

    result = ml_agent.run_full_ml_pipeline(
        df,
        target_col=payload.target_col,
        problem_type=payload.problem_type,
        extra_models=payload.extra_models,
        selected_metrics=payload.selected_metrics,
        user_preprocessing_choices=payload.user_preprocessing_choices,
    )

    orchestrator.update_context("ml_result", result)
    orchestrator.update_context("target_col", payload.target_col)
    orchestrator.update_context("problem_type", payload.problem_type)
    orchestrator.update_context("leaderboard", result.get("leaderboard", []))
    orchestrator.update_context("best_model", result.get("best_model_name"))
    orchestrator.update_context("preprocessing_log", result.get("preprocessing_log", []))
    orchestrator.update_context("model_reasoning", result.get("model_reasoning", {}))

    if result.get("shap_analysis"):
        orchestrator.update_context("shap_result", result.get("shap_analysis"))
        orchestrator.update_context("feature_importance", result.get("shap_analysis", {}).get("feature_importance", []))
        orchestrator.update_context("shap_explanation", result.get("shap_analysis", {}).get("explanation", ""))

    if payload.session_id:
        save_ml_result(db, payload.session_id, result)

        shap_result = result.get("shap_analysis")
        if shap_result:
            save_shap_result(db, payload.session_id, shap_result)
            for chart_type, chart_path in (shap_result.get("charts") or {}).items():
                save_file_record(db, payload.session_id, chart_type, chart_path)

        update_session_status(db, payload.session_id, "trained")

        for file_type, file_path in (result.get("saved_files") or {}).items():
            save_file_record(db, payload.session_id, file_type, file_path)
        if result.get("ml_code_path"):
            save_file_record(db, payload.session_id, "ml_code", result["ml_code_path"])

    return clean_for_json(result)


class ReportRequest(BaseModel):
    filename: str
    target_col: str
    problem_type: str
    session_id: Optional[str] = None


@app.post("/report")
def report_file(payload: ReportRequest, db: DBSession = Depends(get_db)):
    filepath = os.path.join(UPLOAD_FOLDER, payload.filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    if payload.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{payload.target_col}' not found in dataset.")

    analysis_result = analysis_agent.run_full_analysis(df, confirmed_target=payload.target_col)
    ml_result = ml_agent.run_full_ml_pipeline(
        df, target_col=payload.target_col, problem_type=payload.problem_type
    )
    result = report_generator.generate_full_report(analysis_result, ml_result)

    if payload.session_id:
        save_file_record(db, payload.session_id, "report", result.get("report_path"))
        update_session_status(db, payload.session_id, "completed")

    return clean_for_json(result)


DOWNLOAD_FILES = {
    "report": ("report.html", "text/html"),
    "model": ("best_model.pkl", "application/octet-stream"),
    "eda-code": ("eda_report.py", "text/x-python"),
    "preprocessing-code": ("preprocessing.py", "text/x-python"),
    "model-code": ("model_training.py", "text/x-python"),
    "cleaned-data": ("preprocessed_data.csv", "text/csv"),
}


@app.get("/download/{item}")
def download_item(item: str):
    if item not in DOWNLOAD_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown download item '{item}'.")

    filename, media_type = DOWNLOAD_FILES[item]
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.isfile(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' does not exist yet. Run the corresponding pipeline step first.",
        )

    return FileResponse(filepath, filename=filename, media_type=media_type)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/chat")
def chat(payload: ChatRequest, db: DBSession = Depends(get_db)):
    if payload.session_id:
        history = get_chat_history(db, payload.session_id)
        orchestrator.session_context["chat_history"] = [
            {"role": m.role, "content": m.message} for m in history
        ]

    response = orchestrator.chat(payload.message)

    if payload.session_id:
        save_chat_message(db, payload.session_id, "user", payload.message)
        save_chat_message(db, payload.session_id, "assistant", response)

    return {"response": response}


class ContextUpdateRequest(BaseModel):
    key: str
    value: Any


@app.post("/context/update")
def update_context(payload: ContextUpdateRequest):
    orchestrator.update_context(payload.key, payload.value)
    return {"status": "updated"}


def _analysis_result_to_dict(a):
    return {
        "profile": {
            "quality_score": a.quality_score,
            "rows": a.rows,
            "columns": a.columns,
            "column_types": a.column_types,
            "missing": a.missing_values,
        },
        "problem_type": {"problem_type": a.problem_type},
        "target_suggestion": {"suggested_target": a.target_column},
        "charts": a.chart_paths,
        "outliers": a.outliers,
        "target_balance": a.target_balance,
        "insights": a.insights,
        "status": "completed",
    }


def _ml_result_to_dict(m):
    return {
        "best_model_name": m.best_model_name,
        "best_params": m.best_params,
        "leaderboard": m.leaderboard,
        "preprocessing_log": m.preprocessing_log,
        "model_reasoning": m.model_reasoning,
        "tuning_score": m.tuning_score,
        "status": "completed",
    }


def _shap_result_to_dict(s):
    return {
        "status": s.status,
        "feature_importance": s.feature_importance,
        "explanation": s.explanation,
        "charts": s.chart_paths,
    }


@app.get("/sessions")
def list_sessions(db: DBSession = Depends(get_db)):
    sessions = get_all_sessions(db, limit=10)
    result = []
    for s in sessions:
        ml_result = get_ml_result(db, s.session_id)
        result.append({
            "session_id": s.session_id,
            "filename": s.filename,
            "status": s.status,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "best_model": ml_result.best_model_name if ml_result else None,
            "accuracy": ml_result.best_accuracy if ml_result else None,
        })
    return clean_for_json(result)


@app.get("/sessions/{session_id}")
def get_session_detail(session_id: str, db: DBSession = Depends(get_db)):
    analysis = get_analysis_result(db, session_id)
    ml_result = get_ml_result(db, session_id)
    shap_result = get_shap_result(db, session_id)
    files = get_session_files(db, session_id)
    chat_history = get_chat_history(db, session_id, limit=50)

    return clean_for_json({
        "session_id": session_id,
        "analysis_result": _analysis_result_to_dict(analysis) if analysis else None,
        "ml_result": _ml_result_to_dict(ml_result) if ml_result else None,
        "shap_result": _shap_result_to_dict(shap_result) if shap_result else None,
        "files": [{"file_type": f.file_type, "file_path": f.file_path} for f in files],
        "chat_history": [
            {"role": m.role, "message": m.message, "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in chat_history
        ],
    })


@app.get("/sessions/{session_id}/files")
def get_session_download_files(session_id: str, db: DBSession = Depends(get_db)):
    files = get_session_files(db, session_id)
    return clean_for_json([{"file_type": f.file_type, "file_path": f.file_path} for f in files])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
