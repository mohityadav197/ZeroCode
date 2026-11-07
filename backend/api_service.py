"""Compatibility helpers used by the original endpoints and new stage routes."""
import asyncio
from pathlib import Path
from hashlib import sha256

from fastapi import HTTPException

from blackboard import Blackboard
from config import UPLOAD_FOLDER
from pipeline import PipelineOrchestrator
from runtime_context import safe_output_path
from schemas import AgentInput


def dataset_path(filename):
    try:
        path = safe_output_path(str(Path(UPLOAD_FOLDER) / filename))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if not path.is_file():
        raise HTTPException(404, "Dataset or artifact not found")
    return str(path)


def make_request(filename, session_id=None, **options):
    path = dataset_path(filename)
    relative = Path(path).relative_to(Path(UPLOAD_FOLDER).resolve())
    inferred = relative.parts[0] if len(relative.parts) > 1 else "legacy-" + sha256(str(Path(path)).encode()).hexdigest()[:16]
    sid = session_id or inferred
    old = Blackboard(sid).get_full_state()["request"]
    values = {**old, "session_id": sid, "filepath": path, "predecessors": {}}
    values.update({key: value for key, value in options.items() if value is not None})
    return AgentInput.model_validate(values)


def execute_stages(request, stages, pipeline_mode=None, messaging_mode=None):
    options = {}
    if pipeline_mode:
        options["pipeline_mode"] = pipeline_mode
    if messaging_mode:
        options["messaging_mode"] = messaging_mode
    return asyncio.run(PipelineOrchestrator(**options).run(request, stages))


def persist_state(state):
    """Keep the original history/download views current for the new pipeline API."""
    from database import SessionLocal, Session
    from db_service import save_analysis_result, save_ml_result, save_shap_result, save_file_record, update_session_status
    sid = state["session_id"]
    with SessionLocal() as db:
        # Stage APIs operate on uploaded sessions; direct SDK runs can use just a blackboard.
        if db.get(Session, sid) is None:
            return
        status = "uploaded"
        if state["eda_result"]:
            save_analysis_result(db, sid, state["eda_result"])
            status = "analyzed"
            for kind, path in state["eda_result"]["charts"].get("charts", {}).items():
                save_file_record(db, sid, f"chart_{kind}", path)
        if state["ml_result"]:
            ml = state["ml_result"]
            save_ml_result(db, sid, ml)
            status = "trained"
            if ml.get("shap_analysis"):
                save_shap_result(db, sid, ml["shap_analysis"])
            for kind, path in ml.get("saved_files", {}).items():
                save_file_record(db, sid, kind, path)
        if state["report_result"]:
            save_file_record(db, sid, "report", state["report_result"]["report_path"])
            status = "completed"
        update_session_status(db, sid, status)


def analyze(filename, target=None, session_id=None):
    request = make_request(filename, session_id, target_col=target)
    return execute_stages(request, ("profiler", "eda"))["eda_result"]


def train(filename, target_col, problem_type, session_id=None, **choices):
    request = make_request(filename, session_id, target_col=target_col, problem_type=problem_type, **choices)
    state = Blackboard(request.session_id).get_full_state()
    profiler = state["profiler_result"]
    matches = (profiler and profiler["target_suggestion"]["suggested_target"] == target_col
               and profiler["problem_type"]["problem_type"] == problem_type)
    stages = ("preprocessing", "ml") if matches else ("profiler", "eda", "preprocessing", "ml")
    return execute_stages(request, stages)["ml_result"]


def report(filename, target_col, problem_type, session_id=None):
    request = make_request(filename, session_id, target_col=target_col, problem_type=problem_type)
    state = Blackboard(request.session_id).get_full_state()
    old = state["request"]
    ml = state["ml_result"]
    if not ml or ml.get("status") != "completed" or not ml.get("best_model_name") or not ml.get("leaderboard"):
        raise HTTPException(400, "Please train a model before generating a report")
    if old.get("target_col") != target_col or old.get("problem_type") != problem_type:
        raise HTTPException(400, "Please train a model for the selected target and problem type before generating a report")
    if not (Path(UPLOAD_FOLDER) / request.session_id / "best_model.pkl").is_file():
        raise HTTPException(400, "Trained model file is missing. Please train a model before generating a report")
    missing = [stage for stage in ("profiler", "eda", "preprocessing") if not state[f"{stage}_result"]]
    if missing:
        raise HTTPException(400, f"Missing report inputs: {', '.join(missing)}. Please run the full pipeline before generating a report")
    # A report consumes existing results; it must never silently retrain models.
    return execute_stages(request, ("report",))["report_result"]


def session_orchestrator(session_id, db=None):
    from agents.orchestrator import Orchestrator
    from db_service import get_analysis_result, get_ml_result, get_shap_result
    if not session_id:
        from app import orchestrator
        return orchestrator
    agent = Orchestrator()
    board = Blackboard(session_id).get_fields("request", "eda_result", "ml_result")
    request = board["request"]
    analysis_context = board["eda_result"] or {}
    for key, value in {"filename": Path(request["filepath"]).name if request.get("filepath") else None,
                       "target_col": request.get("target_col") or analysis_context.get("target_suggestion", {}).get("suggested_target"),
                       "problem_type": request.get("problem_type") or analysis_context.get("problem_type", {}).get("problem_type"),
                       "analysis_result": board["eda_result"],
                       "ml_result": board["ml_result"]}.items():
        agent.update_context(key, value)
    if board["ml_result"] and board["ml_result"].get("shap_analysis"):
        agent.update_context("shap_result", board["ml_result"]["shap_analysis"])
    if db is not None:
        analysis = get_analysis_result(db, session_id)
        ml = get_ml_result(db, session_id)
        shap = get_shap_result(db, session_id)
        if analysis and not board["eda_result"]:
            from app import _analysis_result_to_dict
            agent.update_context("analysis_result", _analysis_result_to_dict(analysis))
            agent.update_context("target_col", analysis.target_column)
            agent.update_context("problem_type", analysis.problem_type)
        if ml and not board["ml_result"]:
            from app import _ml_result_to_dict
            agent.update_context("ml_result", _ml_result_to_dict(ml))
        if shap:
            agent.update_context("shap_result", {"feature_importance": shap.feature_importance, "explanation": shap.explanation})
    return agent
