from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from app.models.document_models import (
    JobProfile,
    ResumeProfile,
)


load_dotenv()


class BulletSuggestion(BaseModel):
    original_evidence: str
    suggested_rewrite: str
    target_requirement: str
    reason: str


class ResumeOptimizationReport(BaseModel):
    professional_summary_suggestion: str | None = None

    bullet_suggestions: list[BulletSuggestion] = Field(
        default_factory=list
    )

    keywords_to_emphasize: list[str] = Field(
        default_factory=list
    )

    missing_but_unverified_skills: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )


class ResumeOptimizationError(Exception):
    pass


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ResumeOptimizationError(
            "OPENAI_API_KEY is missing."
        )

    return OpenAI(api_key=api_key)


def optimize_resume(
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
) -> ResumeOptimizationReport:

    client = get_client()

    resume_json = resume_profile.model_dump_json(
        indent=2
    )

    job_json = job_profile.model_dump_json(
        indent=2
    )

    prompt = f"""
RESUME PROFILE

{resume_json}


JOB PROFILE

{job_json}
"""

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
                        "You are a resume optimization assistant. "
                        "Improve resume wording only when the candidate's "
                        "existing resume evidence supports the statement. "
                        "Never invent skills, tools, employers, projects, "
                        "metrics, responsibilities, certifications, dates, "
                        "or achievements. "
                        "Do not convert related experience into direct "
                        "experience. "
                        "If the job requests something that is absent from "
                        "the resume, put it in missing_but_unverified_skills "
                        "instead of adding it to the resume. "
                        "Keep suggested bullets concise, professional, "
                        "specific, and ATS-friendly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=ResumeOptimizationReport,
        )

    except Exception as exc:
        raise ResumeOptimizationError(
            f"Resume optimization failed: {exc}"
        ) from exc

    if response.output_parsed is None:
        raise ResumeOptimizationError(
            "No optimization report was returned."
        )

    return response.output_parsed
