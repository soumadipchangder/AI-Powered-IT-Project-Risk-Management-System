# Task Proposals from Codebase Review

## 1) Typo Fix Task
**Task:** Fix markdown heading typos in `README.md` where headings are missing a space after `###` (e.g., `###✅ 3. Set Environment Variables` and `###✅ 4. Run App`).

**Why:** This is a formatting typo that breaks standard markdown heading rendering and readability.

---

## 2) Bug Fix Task
**Task:** Fix metrics extraction in `workflow.py`'s `generate_report` node so it derives metrics from the actual report schema returned by `ReportingAgent`.

**Why:** `generate_report` currently reads `report_result["risk_assessment"]`, `report_result["project_status"]`, and `report_result["market_analysis"]`, but the reporting prompt defines top-level keys like `executive_summary`, `detailed_analysis`, `recommendations`, and `action_items`. As a result, fallback defaults (`0`, `UNKNOWN`, `NEUTRAL`) are likely used most of the time.

---

## 3) Comment/Documentation Discrepancy Task
**Task:** Update `README.md` to match the implemented stack and run command:
- Replace Streamlit references with Gradio.
- Replace `streamlit run app.py` with `python app.py` (or the preferred Gradio launch command).
- Remove or correct references to Mistral/CrewAI if they are not used in code.

**Why:** The README currently documents a different app framework and stack than the codebase, which can mislead contributors and users.

---

## 4) Test Improvement Task
**Task:** Add a test module (e.g., `tests/test_workflow_metrics.py`) that mocks `ReportingAgent.generate_report` and validates `workflow.generate_report` produces correct non-default `metrics` values from report output.

**Why:** There are currently no automated tests, and the metrics mapping bug is exactly the type of regression that a focused unit test can catch early.
