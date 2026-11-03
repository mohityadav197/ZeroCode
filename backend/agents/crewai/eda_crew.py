import pandas as pd

from agents.adk.profiler_agent import describe_frame
from agents.analysis_agent import AnalysisAgent
from agents.crewai.common import run_two_agent_crew
from agents.framework_models import parse_csv
from agents.worker import WorkerNode


class EDACrew(WorkerNode):
    stage = "eda"

    def execute(self, request):
        parsed = parse_csv(request.filepath)
        frame = pd.read_csv(parsed["filepath"])
        # In parallel mode EDA has no dependency on the profiler's completion.
        facts = self.predecessor(request, "profiler") or describe_frame(frame, request.target_col, request.problem_type)
        facts = {key: value for key, value in facts.items() if key not in ("summary", "status")}
        analyst = AnalysisAgent()
        target = facts["target_suggestion"]["suggested_target"]
        outliers = analyst.detect_outliers(frame, facts["profile"])
        balance = analyst.check_target_balance(frame, target) if facts["problem_type"]["problem_type"] != "regression" else {
            "distribution": {}, "is_balanced": True, "warning": None,
        }
        insight = run_two_agent_crew(
            [("StatisticsAnalyst", "Check missingness, outliers and class balance"),
             ("InsightWriter", "Explain the verified findings in plain English")],
            {**facts, "outliers": outliers, "target_balance": balance, "prompt": request.prompt},
            "Analyze these computed statistics and the user's analysis focus. Verify every number.",
            "Write three short plain-English insights. Use only verified statistics from the preceding task.",
        )
        return {**facts, "outliers": outliers, "target_balance": balance, "insights": [insight],
                "charts": analyst.generate_charts(frame, facts["profile"]),
                "eda_code_path": analyst.generate_eda_code(frame, facts["profile"])}
