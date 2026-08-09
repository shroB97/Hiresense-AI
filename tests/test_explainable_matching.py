from app.ats.explainable_matching_engine import (
    create_explainable_match_report,
)
from app.extractors.structured_extractor import (
    extract_job_profile,
    extract_resume_profile,
)


resume_text = """
Alex Morgan
Senior Business Analyst

Facilitated workshops with Finance, Operations,
Technology, and senior business stakeholders.

Built Power BI dashboards for executive reporting.

Migrated business process models from ARIS
to SAP Signavio.

Led a cross-functional process redesign initiative.

Completed Introduction to Lean Six Sigma.
"""


job_text = """
Senior Business Process Analyst

Required:
Strong stakeholder management.
Experience with Power BI.
Experience with SAP Signavio.
Leadership of cross-functional initiatives.

Preferred:
Lean Six Sigma certification.
"""


resume_profile = extract_resume_profile(
    resume_text
)

job_profile = extract_job_profile(
    job_text
)

report = create_explainable_match_report(
    resume_profile=resume_profile,
    job_profile=job_profile,
)

print(
    report.model_dump_json(
        indent=2
    )
)
