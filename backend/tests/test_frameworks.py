from agents.framework_models import run_adk
from agents.crewai.common import run_two_agent_crew
from tool_registry import TOOLS


def test_real_adk_runner_and_crewai_kickoff_use_registry(monkeypatch):
    calls = []

    def fake_llm(prompt, system=""):
        calls.append(prompt)
        return "Final Answer: The dataset has 64 rows. Review class imbalance."

    monkeypatch.setattr(TOOLS["call_llm"], "handler", fake_llm)
    assert "64 rows" in run_adk("test_profiler", "Summarize computed facts", {"rows": 64}, [])
    output = run_two_agent_crew(
        [("StatisticsAnalyst", "Check statistics"), ("InsightWriter", "Explain statistics")],
        {"rows": 64}, "Verify row count", "Summarize the verified count",
    )
    assert "64 rows" in output
    assert len(calls) == 3
