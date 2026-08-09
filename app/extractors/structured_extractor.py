from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from app.models.document_models import (
    JobProfile,
    ResumeProfile,
)


load_dotenv()


class StructuredExtractionError(Exception):
    """Raised when résumé or job extraction fails."""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise StructuredExtractionError(
            "OPENAI_API_KEY is missing from the .env file."
        )

    return OpenAI(api_key=api_key)


def extract_resume_profile(
    resume_text: str,
) -> ResumeProfile:
    if not resume_text.strip():
        raise StructuredExtractionError(
            "The extracted résumé text is empty."
        )

    client = get_client()

    try:
        response = client.responses.parse(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-5-mini",
            ),
            input=[
                {
                    "role": "developer",
                    "content": (
                        "Extract structured information from the résumé. "
                        "Use only information explicitly supported by the "
                        "résumé text. Do not invent experience, tools, skills, "
                        "education, dates, companies, achievements, or "
                        "certifications. Preserve industry-specific wording. "
                        "When information is unavailable, return null or an "
                        "empty list. Put ambiguities into extraction_warnings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract the candidate profile from this résumé:\n\n"
                        + resume_text
                    ),
                },
            ],
            text_format=ResumeProfile,
        )

    except Exception as exc:
        raise StructuredExtractionError(
            f"Résumé extraction failed: {exc}"
        ) from exc

    if response.output_parsed is None:
        raise StructuredExtractionError(
            "No structured résumé profile was returned."
        )

    return response.output_parsed


def extract_job_profile(
    job_description: str,
) -> JobProfile:
    if not job_description.strip():
        raise StructuredExtractionError(
            "The job description is empty."
        )

    client = get_client()

    try:
        response = client.responses.parse(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-5-mini",
            ),
            input=[
                {
                   "role": "developer",
                   "content": (
        "Extract hiring requirements from the job posting. "
        "Create atomic requirements: each requirement should represent "
        "one independently assessable capability, qualification, tool, "
        "responsibility, competency, methodology, domain, certification, "
        "education requirement, or experience requirement. "
        ""
        "Examples: if the posting says 'Strong communication, stakeholder "
        "management, and leadership skills', create three separate "
        "requirements rather than one combined sentence. "
        ""
        "If the posting says 'Experience with Power BI or Tableau', preserve "
        "the alternative relationship rather than inventing requirements. "
        ""
        "Categorize each requirement as technical_skill, tool, certification, "
        "soft_skill, responsibility, experience, education, domain, "
        "methodology, leadership, business_impact, or other. "
        ""
        "Mark required versus preferred only when the posting supports it. "
        "Do not invent requirements."
    ),
},
                {
                    "role": "user",
                    "content": (
                        "Extract the job profile from this posting:\n\n"
                        + job_description
                    ),
                },
            ],
            text_format=JobProfile,
        )

    except Exception as exc:
        raise StructuredExtractionError(
            f"Job-description extraction failed: {exc}"
        ) from exc

    if response.output_parsed is None:
        raise StructuredExtractionError(
            "No structured job profile was returned."
        )

    return response.output_parsed
