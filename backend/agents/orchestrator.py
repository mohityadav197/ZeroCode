# ============================================
# ZeroCode — orchestrator.py
# Orchestrator — receives user chat messages,
# understands intent, routes to correct context,
# and returns intelligent answers using LLM.
# ============================================

from llm_provider import LLMProvider

RECOMMENDATION_KEYWORDS = ["improve", "recommend", "suggest", "next", "what should"]

INTENT_KEYWORDS = {
    "data_question": ["rows", "columns", "shape", "size"],
    "eda_question": ["insight", "insights", "finding", "findings"],
    "model_question": ["model", "accuracy", "f1", "train", "svm", "forest", "predict"],
    "preprocessing_question": ["preprocess", "clean", "encode", "drop"],
    "chart_question": [
        "chart", "histogram", "heatmap", "boxplot", "distribution",
        "correlation", "bar chart", "visualisation", "visualization",
        "plot", "graph", "show me",
    ],
    "shap_question": [
        "shap", "feature importance", "why did", "which feature",
        "what matters", "feature impact", "explain prediction", "why model",
    ],
    "outlier_question": ["outlier", "anomaly", "extreme values", "unusual"],
    "missing_question": ["missing", "null", "empty", "nan", "filled"],
}


def _describe_chart(name: str) -> str:
    if name == "correlation_heatmap":
        return f"{name} shows relationships between all numeric features"
    if name.startswith("histogram_"):
        col = name[len("histogram_"):]
        return f"{name} shows distribution of {col} column"
    if name.startswith("boxplot_"):
        col = name[len("boxplot_"):]
        return f"{name} shows spread and outliers of {col} column"
    if name.startswith("bar_"):
        col = name[len("bar_"):]
        return f"{name} shows frequency of each {col} category"
    return f"{name} is a chart generated during analysis"


class Orchestrator:
    def __init__(self):
        self.llm = LLMProvider()
        self.session_context = {
            "filename": None,
            "target_col": None,
            "problem_type": None,

            # Full analysis data
            "analysis_result": None,
            "domain_info": None,
            "quality_score": None,
            "insights": [],
            "column_types": {},
            "missing_values": {},
            "outliers": {},
            "target_balance": {},
            "charts_generated": [],

            # Full ML data
            "ml_result": None,
            "leaderboard": [],
            "best_model": None,
            "best_accuracy": None,
            "best_f1": None,
            "best_params": {},
            "preprocessing_log": [],
            "model_reasoning": {},

            # SHAP data
            "shap_result": None,
            "feature_importance": [],
            "shap_explanation": "",

            # Chat history
            "chat_history": [],
        }

    # ------------------------------------------------------------------
    def update_context(self, key, value):
        self.session_context[key] = value

        if key == "analysis_result" and value:
            profile = value.get("profile", {}) or {}
            self.session_context["quality_score"] = profile.get("quality_score")
            self.session_context["insights"] = value.get("insights", [])
            self.session_context["column_types"] = profile.get("column_types", {})
            self.session_context["missing_values"] = profile.get("missing", {})
            self.session_context["outliers"] = (value.get("outliers") or {}).get("outliers", {})
            self.session_context["target_balance"] = value.get("target_balance", {})
            charts = (value.get("charts") or {}).get("charts", {})
            self.session_context["charts_generated"] = list(charts.keys())

        elif key == "ml_result" and value:
            leaderboard = value.get("leaderboard", []) or []
            self.session_context["leaderboard"] = leaderboard
            self.session_context["best_model"] = value.get("best_model_name")
            best_entry = next((r for r in leaderboard if r.get("rank") == 1), leaderboard[0] if leaderboard else {})
            self.session_context["best_accuracy"] = best_entry.get("accuracy")
            self.session_context["best_f1"] = best_entry.get("f1")
            self.session_context["best_params"] = value.get("best_params", {})
            self.session_context["preprocessing_log"] = value.get("preprocessing_log", [])
            self.session_context["model_reasoning"] = value.get("model_reasoning", {})

        elif key == "shap_result" and value:
            self.session_context["feature_importance"] = value.get("feature_importance", [])
            self.session_context["shap_explanation"] = value.get("explanation", "")

    # ------------------------------------------------------------------
    def detect_intent(self, message: str) -> str:
        msg = message.lower()

        if any(keyword in msg for keyword in RECOMMENDATION_KEYWORDS):
            return "recommendation_question"

        scores = {
            intent: sum(1 for keyword in keywords if keyword in msg)
            for intent, keywords in INTENT_KEYWORDS.items()
        }
        best_intent = max(scores, key=scores.get)

        if scores[best_intent] == 0:
            return "general"

        return best_intent

    # ------------------------------------------------------------------
    def build_context_prompt(self, intent: str) -> str:
        domain_info = self.session_context.get("domain_info")
        domain_line = ""
        if domain_info:
            domain_line = (
                f"Dataset Domain: {domain_info.get('display_name')} "
                f"({domain_info.get('confidence')} confidence)\n\n"
            )
        return domain_line + self._build_intent_prompt(intent)

    # ------------------------------------------------------------------
    def _build_intent_prompt(self, intent: str) -> str:
        ctx = self.session_context
        analysis_result = ctx.get("analysis_result")
        ml_result = ctx.get("ml_result")

        if intent == "data_question" and analysis_result:
            profile = analysis_result.get("profile", {})
            return (
                f"Dataset: {ctx.get('filename')}\n"
                f"Rows: {profile.get('rows')}, Columns: {profile.get('columns')}\n"
                f"Quality Score: {ctx.get('quality_score')}/100\n"
                f"Column types: {ctx.get('column_types')}\n"
                f"Missing values: {ctx.get('missing_values')}\n"
                f"ID-like columns detected: {profile.get('id_like_columns')}\n"
                f"Target column: {ctx.get('target_col')}\n"
                f"Problem type: {ctx.get('problem_type')}"
            )

        if intent == "chart_question" and ctx.get("charts_generated"):
            charts = ctx["charts_generated"]
            descriptions = "\n".join(f"- {_describe_chart(name)}" for name in charts)
            return (
                f"Charts generated in this session:\n{descriptions}\n"
                f"Total charts: {len(charts)}"
            )

        if intent == "shap_question" and ctx.get("feature_importance"):
            features = "\n".join(
                f"- Rank {f.get('rank')}: {f.get('feature')} (importance: {f.get('importance')})"
                for f in ctx["feature_importance"]
            )
            return (
                "SHAP Analysis Results:\n"
                f"Top features by importance:\n{features}\n"
                f"Explanation: {ctx.get('shap_explanation')}\n"
                "SHAP charts generated:\n"
                "- shap_importance: overall feature importance\n"
                "- shap_summary: impact of each feature per sample\n"
                "- shap_waterfall: why model predicted sample 1"
            )

        if intent == "outlier_question" and ctx.get("outliers"):
            outliers = ctx["outliers"]
            with_outliers = [
                f"{col}: {info.get('count')} outliers ({info.get('percentage')}%)"
                for col, info in outliers.items()
                if info.get("count", 0) > 0
            ]
            without_outliers = [col for col, info in outliers.items() if info.get("count", 0) == 0]
            return (
                "Outliers detected:\n"
                + ("\n".join(with_outliers) if with_outliers else "None found")
                + f"\nColumns with no outliers: {without_outliers}"
            )

        if intent == "missing_question" and ctx.get("missing_values") is not None:
            missing = ctx["missing_values"]
            missing_lines = "\n".join(f"{col}: {pct}% missing" for col, pct in missing.items())
            handling_lines = [
                line for line in ctx.get("preprocessing_log", [])
                if "missing" in line.lower() or "filled" in line.lower() or "dropped" in line.lower()
            ]
            return (
                f"Missing values in dataset:\n{missing_lines}\n"
                f"How they were handled:\n{handling_lines}"
            )

        if intent == "eda_question" and analysis_result:
            return (
                f"EDA Insights found:\n{ctx.get('insights')}\n"
                f"Target balance: {ctx.get('target_balance')}\n"
                f"Charts generated: {ctx.get('charts_generated')}\n"
                f"Correlation info from profile: see insights above for correlation findings"
            )

        if intent == "model_question" and ml_result:
            return (
                "Model Training Results:\n"
                f"Problem type: {ctx.get('problem_type')}\n"
                f"Models trained and ranked:\n{ctx.get('leaderboard')}\n"
                f"Best model: {ctx.get('best_model')}\n"
                f"Accuracy: {ctx.get('best_accuracy')}\n"
                f"F1: {ctx.get('best_f1')}\n"
                f"Best parameters after tuning: {ctx.get('best_params')}\n"
                f"Why each model was selected:\n{ctx.get('model_reasoning')}"
            )

        if intent == "preprocessing_question" and ml_result:
            return f"Preprocessing steps: {ctx.get('preprocessing_log')}"

        if intent == "recommendation_question" and (analysis_result or ml_result):
            profile = (analysis_result or {}).get("profile", {})
            return (
                f"Quality score: {ctx.get('quality_score')}\n"
                f"Best model accuracy: {ctx.get('best_accuracy')}\n"
                f"Dataset size: {profile.get('rows')} rows"
            )

        return "No analysis has been run yet."

    # ------------------------------------------------------------------
    def chat(self, message: str) -> str:
        self.session_context["chat_history"].append({"role": "user", "content": message})

        intent = self.detect_intent(message)
        context_prompt = self.build_context_prompt(intent)

        history_before_current = self.session_context["chat_history"][:-1]
        last_entries = history_before_current[-6:]
        last_3_messages = (
            "\n".join(f"{entry['role'].capitalize()}: {entry['content']}" for entry in last_entries)
            if last_entries
            else "No previous conversation."
        )

        system_prompt = """You are ZeroCode AI — an expert data science
assistant built into the ZeroCode AutoML platform.

You have COMPLETE access to everything that
happened in this analysis session:
- The dataset details and quality
- All visualisations that were generated
- EDA insights and findings
- Preprocessing steps taken
- ML model training results
- SHAP explainability analysis

Always answer based on the actual data from
this session. Never say you don't have access
to charts or results — you do.

If asked about a chart → describe what it shows
based on the data context you have.
If asked why model made a decision → use SHAP data.
If asked about missing values → use exact percentages.

Keep answers clear, simple and under 100 words
unless asked for detail.
Be conversational and helpful like a data scientist
explaining to a business person.

Only state facts that are present in the context
below. Do not invent preprocessing steps, statistics,
numbers, or techniques that are not given to you —
if the context doesn't specify how something was
done, say what happened at a high level instead of
guessing the method."""

        full_prompt = f"""{system_prompt}

Context about the current dataset and results:
{context_prompt}

Chat history (last 3 exchanges):
{last_3_messages}

User question: {message}

Answer:"""

        response = self.llm.ask(full_prompt)

        self.session_context["chat_history"].append({"role": "assistant", "content": response})

        return response
