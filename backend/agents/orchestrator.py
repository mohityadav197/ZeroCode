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
    "eda_question": ["insight", "chart", "correlation", "outlier", "distribution"],
    "model_question": ["model", "accuracy", "f1", "train", "svm", "forest", "predict"],
    "preprocessing_question": ["preprocess", "clean", "encode", "drop", "missing", "null"],
}


class Orchestrator:
    def __init__(self):
        self.llm = LLMProvider()
        self.session_context = {
            "filename": None,
            "target_col": None,
            "problem_type": None,
            "analysis_result": None,
            "ml_result": None,
            "chat_history": [],
        }

    # ------------------------------------------------------------------
    def update_context(self, key, value):
        self.session_context[key] = value

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
        ctx = self.session_context
        analysis_result = ctx.get("analysis_result")
        ml_result = ctx.get("ml_result")

        if intent == "data_question" and analysis_result:
            profile = analysis_result.get("profile", {})
            return (
                f"Rows: {profile.get('rows')}\n"
                f"Columns: {profile.get('columns')}\n"
                f"Column types: {profile.get('column_types')}\n"
                f"Missing values: {profile.get('missing')}\n"
                f"Quality score: {profile.get('quality_score')}/100"
            )

        if intent == "eda_question" and analysis_result:
            return (
                f"Insights: {analysis_result.get('insights')}\n"
                f"Outliers: {analysis_result.get('outliers', {}).get('outliers')}\n"
                f"Target balance: {analysis_result.get('target_balance')}"
            )

        if intent == "model_question" and ml_result:
            return (
                f"Leaderboard: {ml_result.get('leaderboard')}\n"
                f"Best model: {ml_result.get('best_model_name')}\n"
                f"Best params: {ml_result.get('best_params')}"
            )

        if intent == "preprocessing_question" and ml_result:
            return f"Preprocessing steps: {ml_result.get('preprocessing_log')}"

        if intent == "recommendation_question" and (analysis_result or ml_result):
            profile = (analysis_result or {}).get("profile", {})
            leaderboard = (ml_result or {}).get("leaderboard", [])
            best_entry = next((r for r in leaderboard if r.get("is_best")), leaderboard[0] if leaderboard else {})
            best_metric = best_entry.get("accuracy", best_entry.get("r2"))
            return (
                f"Quality score: {profile.get('quality_score')}\n"
                f"Best model accuracy: {best_metric}\n"
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

        full_prompt = f"""You are ZeroCode AI assistant.
You help users understand their data and ML results.
Answer in simple, plain English. Be concise.
Keep answers under 100 words unless the user asks for detail.

Context about the current dataset and results:
{context_prompt}

Chat history (last 3 exchanges):
{last_3_messages}

User question: {message}

Answer:"""

        response = self.llm.ask(full_prompt)

        self.session_context["chat_history"].append({"role": "assistant", "content": response})

        return response
