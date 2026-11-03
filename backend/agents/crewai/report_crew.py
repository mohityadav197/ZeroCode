from html import escape

from agents.crewai.common import run_two_agent_crew
from agents.worker import WorkerNode
from utils.report_generator import ReportGenerator


def escaped_values(value):
    if isinstance(value, str):
        return escape(value)
    if isinstance(value, list):
        return [escaped_values(item) for item in value]
    if isinstance(value, dict):
        return {escape(str(key)): escaped_values(item) for key, item in value.items()}
    return value


class ReportCrew(WorkerNode):
    stage = "report"

    def execute(self, request):
        analysis = self.predecessor(request, "eda")
        ml = self.predecessor(request, "ml")
        profiler = self.predecessor(request, "profiler")
        prep = self.predecessor(request, "preprocessing")
        if not ml or ml.get("status") != "completed" or not ml.get("best_model_name"):
            raise ValueError("Please train a model before generating a report")
        if not all((analysis, profiler, prep)):
            raise ValueError("Report requires completed profiler, EDA, preprocessing and ML results")
        generator = ReportGenerator()
        verification = generator.verify_data(analysis, ml)
        if analysis["profile"] != profiler["profile"]:
            verification["issues"].append("Profiler and EDA statistics disagree")
        if ml["preprocessing_log"] != prep["preprocessing_log"]:
            verification["issues"].append("ML preprocessing differs from preprocessing worker")
        if not any(row.get("model") == ml["best_model_name"] or row.get("model_name") == ml["best_model_name"]
                   for row in ml["leaderboard"]):
            verification["issues"].append("Best model is absent from the leaderboard")
        for row in ml["leaderboard"]:
            for metric in ("accuracy", "f1", "precision", "recall", "roc_auc", "balanced_accuracy"):
                value = row.get(metric)
                if value is not None and not 0 <= value <= 1:
                    verification["issues"].append(f"Invalid {metric} for {row.get('model')}")
        verification["verified"] = not verification["issues"]
        verification["stamp"] = "Verified" if verification["verified"] else "Verification failed"
        try:
            narrative = run_two_agent_crew(
                [("DataVerifier", "Verify consistency of the supplied agent results"),
                 ("ReportWriter", "Write final report recommendations from verified numbers")],
                {"profile": profiler["profile"], "insights": analysis["insights"], "leaderboard": ml["leaderboard"],
                 "preprocessing": prep["preprocessing_log"],
                 "best_model": ml["best_model_name"], "verification": verification, "prompt": request.prompt},
                "Cross-check the numbers and the supplied deterministic verification issues. Do not suppress issues.",
                "Write a short final report narrative and actionable recommendations, using the verified numbers. "
                "This text will be embedded in the final HTML report alongside the full computed tables.",
            )
        except Exception as error:
            raise RuntimeError(f"Report crew failed: {error}") from error
        from guardrails import InputGuard
        InputGuard().validate_llm_response(narrative)
        # Embed images so the downloaded HTML remains usable without a running server.
        import base64
        from pathlib import Path
        from config import UPLOAD_FOLDER
        from runtime_context import safe_output_path
        displayed_analysis = escaped_values(analysis)
        for name, chart_path in (analysis.get("charts") or {}).get("charts", {}).items():
            path = safe_output_path(str(Path(UPLOAD_FOLDER) / chart_path.removeprefix("outputs/")))
            if path.is_file():
                displayed_analysis["charts"]["charts"][escape(name)] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        html = generator.build_html_report(displayed_analysis, escaped_values(ml), [escape(narrative)], verification)
        return {"report_path": generator.save_report(html), "verification": verification, "recommendations": [narrative]}
