# ============================================
# ZeroCode — app.py
# Main FastAPI application — handles all API
# routes, file uploads, and CORS setup.
# ============================================

import os
import io
import asyncio
import logging
import time
import traceback
from typing import Any, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from openai import APIError, RateLimitError

from agents.analysis_agent import AnalysisAgent
from agents.ml_agent import MLAgent
from agents.orchestrator import Orchestrator
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, PROJECT_NAME, UPLOAD_FOLDER
from database import GeneratedFile, get_db, init_db
from db_service import (
    create_session,
    delete_session,
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
import api_service
from blackboard import Blackboard
from guardrails import GuardrailViolation, InputGuard
from observability import setup_tracing
from platform_api import router as platform_router
from runtime_context import execution_context
from schemas import AgentInput
from mcp_tools.mcp_groq import LLM_TIMEOUT_SECONDS, LLMRequestError

# uvicorn only configures its own loggers; without this, INFO from app code is dropped.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="ZeroCode API")
app.include_router(platform_router)
setup_tracing(app)
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


@app.exception_handler(GuardrailViolation)
async def guardrail_error(request, error):
    return JSONResponse(status_code=400, content={"detail": str(error), "guardrail": "blocked"})


@app.exception_handler(ValueError)
async def invalid_input(request, error):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(RuntimeError)
async def pipeline_error(request, error):
    code = 409 if "already running" in str(error) else 500
    return JSONResponse(status_code=code, content={"detail": str(error)})


@app.exception_handler(RateLimitError)
async def llm_rate_limit(request, error):
    retry_after = error.response.headers.get("retry-after", "60")
    return JSONResponse(status_code=429, headers={"Retry-After": retry_after},
                        content={"detail": "Gemini rate limit reached. Retry in a moment.",
                                 "retry_after": retry_after})


@app.exception_handler(APIError)
async def llm_provider_error(request, error):
    return JSONResponse(status_code=502, content={"detail": "Gemini service error. Retrying failed."})


@app.on_event("startup")
def startup():
    init_db()


def error_response(message):
    # Returned from inside the route: an unhandled exception would skip CORSMiddleware,
    # and the browser would hide the body behind a generic "Network Error".
    return JSONResponse(status_code=500, content={"error": True, "message": message})


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
    if not file.filename or os.path.basename(file.filename) != file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(400, "Use a plain CSV filename")
    InputGuard().validate_user_text(file.filename)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only {ALLOWED_EXTENSIONS} files are allowed.",
        )

    contents = await file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    validation = InputGuard().validate_csv_schema(df)
    if not validation.valid:
        raise HTTPException(400, detail="; ".join(validation.errors))
    session = create_session(db, file.filename, len(df), len(df.columns))
    stored_filename = f"{session.session_id}/{file.filename}"
    session.filename = stored_filename
    db.commit()
    filepath = os.path.join(UPLOAD_FOLDER, stored_filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as output:
        output.write(contents)
    Blackboard(session.session_id, AgentInput(session_id=session.session_id, filepath=filepath))
    orchestrator.update_context("session_id", session.session_id)

    return clean_for_json({
        "success": True,
        "session_id": session.session_id,
        "filename": stored_filename,
        "original_filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    })


@app.get("/files/{filename:path}")
def get_file(filename: str):
    filepath = api_service.dataset_path(filename.removeprefix("outputs/"))
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return FileResponse(filepath)


class AnalyzeRequest(BaseModel):
    filename: str
    confirmed_target: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/analyze")
def analyze_file(payload: AnalyzeRequest, db: DBSession = Depends(get_db)):
    try:
        filepath = api_service.dataset_path(payload.filename)
        orchestrator = api_service.session_orchestrator(payload.session_id, db)
        if not os.path.isfile(filepath):
            raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

        result = api_service.analyze(payload.filename, payload.confirmed_target, payload.session_id)

        problem_type_str = (result.get("problem_type", {}) or {}).get("problem_type")
        profile = result.get("profile", {}) or {}
        run_session = api_service.make_request(payload.filename, payload.session_id).session_id
        with execution_context(run_session):
            model_recs = ml_agent.generate_model_recommendations(df, problem_type_str, profile, result)
        result["model_recommendations"] = model_recs
        orchestrator.update_context("pipeline_session_id", run_session)

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
    except HTTPException:
        raise
    except Exception:
        # The exception handlers above answer without logging, which hides the traceback.
        traceback.print_exc()
        raise


class PreprocessingRecommendationsRequest(BaseModel):
    filename: str
    target_col: str
    selected_models: Optional[List[str]] = None


@app.post("/preprocessing-recommendations")
def preprocessing_recommendations(payload: PreprocessingRecommendationsRequest):
    filepath = api_service.dataset_path(payload.filename)
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
    filepath = api_service.dataset_path(payload.filename)
    orchestrator = api_service.session_orchestrator(payload.session_id, db)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    if payload.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{payload.target_col}' not found in dataset.")

    result = api_service.train(
        payload.filename,
        target_col=payload.target_col,
        problem_type=payload.problem_type,
        extra_models=payload.extra_models,
        selected_metrics=payload.selected_metrics,
        user_preprocessing_choices=payload.user_preprocessing_choices,
        session_id=payload.session_id,
    )
    orchestrator.update_context("pipeline_session_id", api_service.make_request(payload.filename, payload.session_id).session_id)

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

    print("Orchestrator context after training:")
    for k, v in orchestrator.session_context.items():
        line = f"  {k}: {str(v)[:100]}"
        print(line.encode("ascii", errors="replace").decode("ascii"))

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
    logger.info("Report request received: %s", payload.model_dump())
    filepath = api_service.dataset_path(payload.filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{payload.filename}' not found.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    if payload.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{payload.target_col}' not found in dataset.")

    try:
        logger.info("Generating report...")
        started = time.perf_counter()
        result = api_service.report(payload.filename, payload.target_col, payload.problem_type, payload.session_id)
        logger.info("Report generated in %.2fs", time.perf_counter() - started)
        if not result or not result.get("report_path") or not os.path.isfile(result["report_path"]):
            raise RuntimeError("Report generation did not produce a report file")
        orchestrator.update_context("pipeline_session_id", api_service.make_request(payload.filename, payload.session_id).session_id)

        if payload.session_id:
            save_file_record(db, payload.session_id, "report", result["report_path"])
            update_session_status(db, payload.session_id, "completed")
    except HTTPException:
        raise
    except Exception as error:
        if "already running" in str(error):
            # Leave this to pipeline_error, which answers 409: the session is busy, not broken.
            logger.warning("Report rejected for session %s: %s", payload.session_id, error)
            raise
        logger.exception("Report failed")
        return error_response(f"Report generation failed: {error}")

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
def download_item(item: str, session_id: Optional[str] = None):
    if item not in DOWNLOAD_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown download item '{item}'.")

    filename, media_type = DOWNLOAD_FILES[item]
    session_id = session_id or orchestrator.session_context.get("pipeline_session_id")
    if session_id:
        AgentInput(session_id=session_id)
        filepath = os.path.join(UPLOAD_FOLDER, session_id, filename)
    else:
        # Keep the existing download URL working for single-session clients.
        from database import Session as SavedSession, SessionLocal
        with SessionLocal() as download_db:
            latest = download_db.query(SavedSession).order_by(SavedSession.created_at.desc()).first()
            filepath = os.path.join(UPLOAD_FOLDER, latest.session_id, filename) if latest else os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.isfile(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' does not exist yet. Run the corresponding pipeline step first.",
        )

    return FileResponse(filepath, filename=filename, media_type=media_type)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


def _load_chat_orchestrator(session_id, db):
    orchestrator = api_service.session_orchestrator(session_id, db)
    print("Session ID in chat:", session_id)

    if session_id:
        history = get_chat_history(db, session_id)
        orchestrator.session_context["chat_history"] = [
            {"role": m.role, "content": m.message} for m in history
        ]

    print("Context at chat time:")
    best_model_line = f"best_model: {orchestrator.session_context.get('best_model')}"
    leaderboard_line = f"leaderboard: {orchestrator.session_context.get('leaderboard')}"
    print(best_model_line.encode("ascii", errors="replace").decode("ascii"))
    print(leaderboard_line.encode("ascii", errors="replace").decode("ascii"))
    return orchestrator


def _save_chat_exchange(session_id, message, response, db):
    if session_id:
        save_chat_message(db, session_id, "user", message)
        save_chat_message(db, session_id, "assistant", response)


@app.post("/chat")
async def chat(payload: ChatRequest, db: DBSession = Depends(get_db)):
    logger.info("Chat request received: %s", payload.model_dump())
    validation = InputGuard().validate_user_text(payload.message)
    if payload.session_id:
        AgentInput(session_id=payload.session_id)

    try:
        # These helpers do blocking database work, which must stay off the event loop.
        orchestrator = await asyncio.to_thread(_load_chat_orchestrator, payload.session_id, db)
        logger.info("Calling LLM...")
        started = time.perf_counter()
        with execution_context(payload.session_id or "legacy"):
            response = await asyncio.wait_for(
                asyncio.to_thread(orchestrator.chat, payload.message), timeout=LLM_TIMEOUT_SECONDS,
            )
        logger.info("LLM responded in %.2fs", time.perf_counter() - started)
        if not isinstance(response, str) or not response.strip():
            raise LLMRequestError("Chat returned an empty response")
        await asyncio.to_thread(_save_chat_exchange, payload.session_id, payload.message, response, db)
    except TimeoutError:
        logger.exception("Chat failed")
        return error_response(f"The LLM did not respond within {LLM_TIMEOUT_SECONDS} seconds")
    except Exception as error:
        logger.exception("Chat failed")
        return error_response(str(error) or type(error).__name__)

    return {"response": response, "warnings": validation.warnings}


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


def _delete_session_data(session_id: str, db: DBSession):
    # outputs/ is a single shared folder reused by every session (each new
    # /train or /analyze run overwrites the same filenames), so a file path
    # recorded for this session may still be "owned" by another session's
    # records. Only delete a physical file if no other session references
    # the same path — otherwise we'd silently destroy another session's
    # (possibly the currently active one's) downloadable artifacts.
    session_files = db.query(GeneratedFile).filter(GeneratedFile.session_id == session_id).all()
    file_paths = {f.file_path for f in session_files if f.file_path}

    other_paths = set()
    if file_paths:
        other_records = (
            db.query(GeneratedFile)
            .filter(GeneratedFile.session_id != session_id, GeneratedFile.file_path.in_(file_paths))
            .all()
        )
        other_paths = {f.file_path for f in other_records}

    deleted = delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    for path in file_paths - other_paths:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass  # best effort — never fail the request over file cleanup

    return {"status": "deleted", "session_id": session_id}


@app.delete("/sessions/{session_id}")
def delete_session_route(session_id: str, db: DBSession = Depends(get_db)):
    from session_locks import session_lock
    with session_lock(session_id):
        return _delete_session_data(session_id, db)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=[os.path.abspath(UPLOAD_FOLDER)])
