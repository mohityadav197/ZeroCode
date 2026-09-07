# ============================================
# ZeroCode — report_generator.py
# Report Generator — collects outputs from
# Analysis and ML agents, verifies data,
# generates HTML report with all charts,
# stats, leaderboard and LLM recommendations.
# ============================================

import os
from datetime import datetime

from config import UPLOAD_FOLDER
from llm_provider import LLMProvider

REPORT_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #f4f5f7;
  color: #1a1a2e;
}
.container { max-width: 1000px; margin: 0 auto; padding: 24px; }
header {
  background: #1a1a2e;
  color: #ffffff;
  padding: 32px 24px;
  border-radius: 10px;
  margin-bottom: 24px;
}
header h1 { margin: 0; font-size: 28px; }
header .subtitle { margin: 4px 0 0; color: #c7c9e0; }
header .timestamp { margin: 8px 0 0; font-size: 13px; color: #9a9cc0; }
.stamp {
  display: inline-block;
  margin-top: 16px;
  padding: 6px 14px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 14px;
}
.stamp.verified { background: #1f4d2e; color: #6dffa0; }
.stamp.warning { background: #4d2c1f; color: #ffb26d; }
.issues {
  background: #fff3cd;
  border: 1px solid #ffe69c;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 24px;
}
section {
  background: #ffffff;
  border-radius: 10px;
  padding: 24px;
  margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
section h2 {
  margin-top: 0;
  color: #1a1a2e;
  border-bottom: 2px solid #eaeaf2;
  padding-bottom: 8px;
}
section h3 { color: #333355; margin-top: 24px; }
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.stat-box {
  flex: 1 1 140px;
  background: #f4f5f7;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.stat-value { display: block; font-size: 24px; font-weight: 700; color: #1a1a2e; }
.stat-label { display: block; font-size: 13px; color: #666; margin-top: 4px; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
}
th, td {
  border: 1px solid #e2e2ec;
  padding: 8px 12px;
  text-align: left;
  font-size: 14px;
}
th { background: #1a1a2e; color: #fff; }
tbody tr:nth-child(even) { background: #f7f8fb; }
tr.best-row {
  background: rgba(40, 167, 69, 0.15) !important;
  font-weight: 700;
  border-left: 4px solid #28a745;
}
.chart-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }
.chart-card {
  flex: 1 1 280px;
  border: 1px solid #eaeaf2;
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.chart-card h4 { margin: 0 0 8px; font-size: 14px; color: #444; }
.chart-card img { max-width: 100%; border-radius: 6px; }
.split-info { font-style: italic; color: #555; }
.ai-note { font-size: 13px; color: #888; margin-top: 16px; }
.warning { color: #b45309; font-weight: 600; }
.ok { color: #1f7a3d; font-weight: 600; }
footer { text-align: center; color: #888; font-size: 13px; padding: 16px 0; }
@media (max-width: 600px) {
  .stat-row { flex-direction: column; }
}
"""


class ReportGenerator:
    def __init__(self):
        self.llm = LLMProvider()

    # ------------------------------------------------------------------
    def verify_data(self, analysis_result: dict, ml_result: dict) -> dict:
        issues = []

        profile = (analysis_result or {}).get("profile")
        if not profile or "rows" not in profile or "columns" not in profile:
            issues.append("Analysis result is missing a valid 'profile' with rows and columns")

        leaderboard = (ml_result or {}).get("leaderboard")
        if not leaderboard or len(leaderboard) < 1:
            issues.append("ML result is missing a leaderboard with at least 1 model")

        if not (ml_result or {}).get("best_model_name"):
            issues.append("ML result is missing a valid best_model_name")

        if not (ml_result or {}).get("preprocessing_log"):
            issues.append("ML result is missing a non-empty preprocessing_log")

        if not (analysis_result or {}).get("insights"):
            issues.append("Analysis result is missing non-empty insights")

        verified = len(issues) == 0
        return {
            "verified": verified,
            "issues": issues,
            "stamp": "Verified ✅" if verified else "Verification failed ⚠️",
        }

    # ------------------------------------------------------------------
    def generate_recommendations(self, analysis_result: dict, ml_result: dict) -> list:
        profile = (analysis_result or {}).get("profile", {})
        target_balance = (analysis_result or {}).get("target_balance", {})
        leaderboard = (ml_result or {}).get("leaderboard", [])
        best_model_name = (ml_result or {}).get("best_model_name")

        best_entry = next((r for r in leaderboard if r.get("is_best")), leaderboard[0] if leaderboard else {})
        if "f1" in best_entry:
            metric_str = f"accuracy {best_entry.get('accuracy')}, f1 {best_entry.get('f1')}"
        elif "r2" in best_entry:
            metric_str = f"r2 {best_entry.get('r2')}, mae {best_entry.get('mae')}"
        else:
            metric_str = "no metrics available"

        is_balanced = target_balance.get("is_balanced", True)
        imbalance_str = target_balance.get("warning") if not is_balanced else "Target classes are balanced"

        prompt = (
            f"Dataset size: {profile.get('rows')} rows, {profile.get('columns')} columns\n"
            f"Quality score: {profile.get('quality_score')}/100\n"
            f"Best model: {best_model_name} ({metric_str})\n"
            f"Target imbalance: {imbalance_str}\n"
            f"Analysis warnings: {'; '.join((analysis_result or {}).get('insights', [])[:3])}\n\n"
            "Based on this ML pipeline summary, give 3 to 5 short, actionable, plain-English "
            "recommendations for improving the model or the data."
        )

        system_prompt = (
            'Respond with JSON in exactly this shape: {"recommendations": ["...", "..."]}. '
            "Each recommendation should be one concise sentence."
        )

        result = self.llm.ask_json(prompt, system_prompt=system_prompt)
        recommendations = result.get("recommendations") if isinstance(result, dict) else None

        if recommendations:
            return recommendations

        return [
            f"Best model achieved is {best_model_name}; validate on more data before production use.",
            "Review the preprocessing log and feature set for opportunities to improve performance.",
        ]

    # ------------------------------------------------------------------
    def build_html_report(self, analysis_result: dict, ml_result: dict, recommendations: list, verification: dict) -> str:
        analysis_result = analysis_result or {}
        ml_result = ml_result or {}
        profile = analysis_result.get("profile", {})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stamp_class = "verified" if verification.get("verified") else "warning"

        header_html = f"""
        <header>
          <h1>ZeroCode</h1>
          <p class="subtitle">AutoML Analysis Report</p>
          <p class="timestamp">Generated on {timestamp}</p>
          <div class="stamp {stamp_class}">{verification.get('stamp')}</div>
        </header>
        """

        issues_html = ""
        if verification.get("issues"):
            issues_items = "".join(f"<li>{issue}</li>" for issue in verification["issues"])
            issues_html = f'<div class="issues"><strong>Issues found:</strong><ul>{issues_items}</ul></div>'

        section1 = f"""
        <section>
          <h2>1. Dataset Overview</h2>
          <div class="stat-row">
            <div class="stat-box"><span class="stat-value">{profile.get('rows', 'N/A')}</span><span class="stat-label">Rows</span></div>
            <div class="stat-box"><span class="stat-value">{profile.get('columns', 'N/A')}</span><span class="stat-label">Columns</span></div>
            <div class="stat-box"><span class="stat-value">{profile.get('quality_score', 'N/A')}/100</span><span class="stat-label">Quality Score</span></div>
          </div>
          <h3>Column Types</h3>
          {self._column_types_table(profile)}
          <h3>Missing Values</h3>
          {self._missing_table(profile)}
        </section>
        """

        charts_result = analysis_result.get("charts", {})
        outliers_result = analysis_result.get("outliers", {})
        target_balance = analysis_result.get("target_balance", {})
        insights = analysis_result.get("insights", [])

        section2 = f"""
        <section>
          <h2>2. EDA Insights</h2>
          {self._insights_list(insights)}
          <h3>Target Balance</h3>
          {self._target_balance_html(target_balance)}
          <h3>Outlier Summary</h3>
          {self._outliers_html(outliers_result)}
          <h3>Charts</h3>
          {self._charts_html(charts_result)}
        </section>
        """

        preprocessing_log = ml_result.get("preprocessing_log", [])
        split_line = next((s for s in preprocessing_log if "Train size" in s), None)

        section3 = f"""
        <section>
          <h2>3. Preprocessing Summary</h2>
          {self._preprocessing_html(preprocessing_log)}
          {f'<p class="split-info">{split_line}</p>' if split_line else ''}
        </section>
        """

        section4 = f"""
        <section>
          <h2>4. Model Results</h2>
          <h3>Model Selection Reasoning</h3>
          {self._model_reasoning_html(ml_result.get('model_reasoning', {}))}
          <h3>Leaderboard</h3>
          {self._leaderboard_html(ml_result.get('leaderboard', []))}
          <h3>Best Model Parameters</h3>
          {self._best_params_html(ml_result.get('best_params', {}), ml_result.get('tuning_score'))}
        </section>
        """

        section5 = f"""
        <section>
          <h2>5. Recommendations</h2>
          {self._recommendations_html(recommendations)}
          <p class="ai-note">Generated by ZeroCode AI</p>
        </section>
        """

        footer_html = f'<footer>ZeroCode AutoML Report &mdash; generated {timestamp}</footer>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZeroCode Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
{header_html}
{issues_html}
{section1}
{section2}
{section3}
{section4}
{section5}
{footer_html}
</div>
</body>
</html>"""

        return html

    # ------------------------------------------------------------------
    @staticmethod
    def _column_types_table(profile: dict) -> str:
        column_types = profile.get("column_types", {})
        if not column_types:
            return "<p>No column type information available.</p>"
        rows = "".join(f"<tr><td>{col}</td><td>{ctype}</td></tr>" for col, ctype in column_types.items())
        return f"<table><thead><tr><th>Column</th><th>Type</th></tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _missing_table(profile: dict) -> str:
        missing = {col: pct for col, pct in profile.get("missing", {}).items() if pct > 0}
        if not missing:
            return "<p>No missing values detected.</p>"
        rows = "".join(f"<tr><td>{col}</td><td>{pct}%</td></tr>" for col, pct in missing.items())
        return f"<table><thead><tr><th>Column</th><th>Missing %</th></tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _insights_list(insights: list) -> str:
        if not insights:
            return "<p>No insights available.</p>"
        items = "".join(f"<li>{insight}</li>" for insight in insights)
        return f"<ul>{items}</ul>"

    @staticmethod
    def _target_balance_html(target_balance: dict) -> str:
        distribution = (target_balance or {}).get("distribution", {})
        if not distribution:
            return "<p>Not applicable for this problem type.</p>"
        rows = "".join(f"<tr><td>{cls}</td><td>{pct}%</td></tr>" for cls, pct in distribution.items())
        table = f"<table><thead><tr><th>Class</th><th>Percentage</th></tr></thead><tbody>{rows}</tbody></table>"
        warning = target_balance.get("warning")
        note = f'<p class="warning">⚠️ {warning}</p>' if warning else '<p class="ok">Target classes are balanced.</p>'
        return table + note

    @staticmethod
    def _outliers_html(outliers_result: dict) -> str:
        outliers = (outliers_result or {}).get("outliers", {})
        rows_data = [(col, info) for col, info in outliers.items() if info.get("count", 0) > 0]
        if not rows_data:
            return "<p>No significant outliers detected.</p>"
        rows = "".join(
            f"<tr><td>{col}</td><td>{info['count']}</td><td>{info['percentage']}%</td></tr>"
            for col, info in rows_data
        )
        return f"<table><thead><tr><th>Column</th><th>Outlier Count</th><th>Percentage</th></tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _charts_html(charts_result: dict) -> str:
        charts = (charts_result or {}).get("charts", {})
        if not charts:
            return "<p>No charts were generated.</p>"
        cards = ""
        for name, path in charts.items():
            rel_path = path[len("outputs/"):] if path.startswith("outputs/") else path
            title = name.replace("_", " ").title()
            cards += f'<div class="chart-card"><h4>{title}</h4><img src="{rel_path}" alt="{title}"></div>'
        return f'<div class="chart-grid">{cards}</div>'

    @staticmethod
    def _preprocessing_html(log: list) -> str:
        if not log:
            return "<p>No preprocessing steps recorded.</p>"
        items = "".join(f"<li>{step}</li>" for step in log)
        return f"<ol>{items}</ol>"

    @staticmethod
    def _model_reasoning_html(reasoning: dict) -> str:
        if not reasoning:
            return "<p>No model reasoning available.</p>"
        items = "".join(f"<li><strong>{name}</strong>: {reason}</li>" for name, reason in reasoning.items())
        return f"<ul>{items}</ul>"

    @staticmethod
    def _leaderboard_html(leaderboard: list) -> str:
        if not leaderboard:
            return "<p>No models were successfully trained.</p>"

        is_regression = "r2" in leaderboard[0]
        if is_regression:
            header = "<tr><th>Rank</th><th>Model</th><th>MAE</th><th>RMSE</th><th>R2</th><th>Time</th></tr>"
        else:
            header = "<tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>Time</th></tr>"

        rows = ""
        for entry in leaderboard:
            row_class = "best-row" if entry.get("is_best") else ""
            badge = " 🏆" if entry.get("is_best") else ""
            if is_regression:
                rows += (
                    f"<tr class='{row_class}'><td>{entry['rank']}</td><td>{entry['model']}{badge}</td>"
                    f"<td>{entry['mae']}</td><td>{entry['rmse']}</td><td>{entry['r2']}</td>"
                    f"<td>{entry['training_time']}</td></tr>"
                )
            else:
                rows += (
                    f"<tr class='{row_class}'><td>{entry['rank']}</td><td>{entry['model']}{badge}</td>"
                    f"<td>{entry['accuracy']}</td><td>{entry['f1']}</td><td>{entry['precision']}</td>"
                    f"<td>{entry['recall']}</td><td>{entry['training_time']}</td></tr>"
                )

        return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _best_params_html(best_params: dict, tuning_score) -> str:
        if not best_params:
            return "<p>No hyperparameter tuning was applied to this model.</p>"
        items = "".join(f"<li><strong>{k}</strong>: {v}</li>" for k, v in best_params.items())
        score_html = f"<p>Tuning CV score: {tuning_score}</p>" if tuning_score is not None else ""
        return f"<ul>{items}</ul>{score_html}"

    @staticmethod
    def _recommendations_html(recommendations: list) -> str:
        if not recommendations:
            return "<p>No recommendations available.</p>"
        items = "".join(f"<li>{rec}</li>" for rec in recommendations)
        return f"<ul>{items}</ul>"

    # ------------------------------------------------------------------
    def save_report(self, html_content: str) -> str:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, "report.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filepath.replace("\\", "/")

    # ------------------------------------------------------------------
    def generate_full_report(self, analysis_result: dict, ml_result: dict) -> dict:
        verification = self.verify_data(analysis_result, ml_result)
        recommendations = self.generate_recommendations(analysis_result, ml_result)
        html_content = self.build_html_report(analysis_result, ml_result, recommendations, verification)
        report_path = self.save_report(html_content)

        return {
            "report_path": report_path,
            "verification": verification,
            "recommendations": recommendations,
            "status": "completed",
        }
