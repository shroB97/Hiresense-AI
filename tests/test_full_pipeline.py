from app.ats.matching_engine import (
    create_match_report,
)
from app.extractors.structured_extractor import (
    extract_job_profile,
    extract_resume_profile,
)


resume_text = """
Jordan Smith
Senior Business Analyst

Five years of experience in business process improvement,
stakeholder engagement, requirements gathering, Power BI,
SQL, SAP Signavio and ARIS.

At Example Consulting, migrated more than 500 process models
from ARIS to Signavio and supported an SAP implementation.
"""

job_description = """
SAP Finance Transformation Analyst

Required:
Experience with stakeholder management, S/4HANA,
Product Costing, Variance Analysis and process design.

Preferred:
JIRA, Confluence and SAP Signavio.
"""


resume_profile = extract_resume_profile(
    resume_text
)

job_profile = extract_job_profile(
    job_description
)

report = create_match_report(
    resume_profile,
    job_profile,
)

print(
    report.model_dump_json(
        indent=2
    )
)
