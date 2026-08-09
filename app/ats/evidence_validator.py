from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from app.models.document_models import JobRequirement


load_dotenv()


class EvidenceCandidate(BaseModel):
    source: str
    text: str


class ValidatedRequirement(BaseModel):
    requirement: str
    category: str
    importance: str

    status: Literal[
        "direct_match",
        "related_evidence",
        "uncertain",
        "not_found",
    ]

    confidence: float = Field(
        ge=0,
        le=100,
    )

    selected_evidence_index: int | None = None

    evidence_text: str | None = None
    evidence_source: str | None = None

    reason: str

    @field_validator(
        "confidence",
        mode="before",
    )
    @classmethod
    def normalize_confidence(
        cls,
        value,
    ):
        value = float(value)

        # If model returns 0.95 instead of 95,
        # convert it automatically.
        if 0 <= value <= 1:
            return value * 100

        return value


class EvidenceValidationError(Exception):
    pass


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EvidenceValidationError(
            "OPENAI_API_KEY is missing."
        )

    return OpenAI(
        api_key=api_key
    )


def validate_requirement(
    requirement: JobRequirement,
    evidence_candidates: list[EvidenceCandidate],
) -> ValidatedRequirement:

    client = get_client()

    evidence_text = "\n\n".join(
        (
            f"Evidence {index}\n"
            f"Source: {candidate.source}\n"
            f"Text: {candidate.text}"
        )
        for index, candidate in enumerate(
            evidence_candidates,
            start=1,
        )
    )

    if not evidence_text:
        evidence_text = (
            "No relevant resume evidence retrieved."
        )

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
                        "You are an evidence-based resume evaluator. "

                        "Assess exactly ONE job requirement against ONLY "
                        "the supplied resume evidence. "

                        "Never invent candidate experience, qualifications, "
                        "skills, credentials, achievements, or evidence. "

                        "Do not require exact wording when evaluating "
                        "experience, responsibilities, soft skills, "
                        "leadership, communication, problem solving, "
                        "stakeholder management, adaptability, ownership, "
                        "or similar competencies. Behavioral evidence can "
                        "demonstrate these requirements. "

                        "Use direct_match only when the supplied resume "
                        "evidence clearly satisfies the requirement. "

                        "Use related_evidence when the resume demonstrates "
                        "a closely related capability but does not fully "
                        "verify the exact requirement. "

                        "Use uncertain when the evidence is weak, incomplete, "
                        "or ambiguous. "

                        "Use not_found when the supplied evidence does not "
                        "support the requirement. "

                        "For named tools and specific technologies, do not "
                        "assume one technology proves another. For example, "
                        "SAP experience alone does not prove SAP S/4HANA. "

                        "For formal certifications, licenses, degrees, and "
                        "credentials, direct_match requires explicit evidence "
                        "that the candidate holds the requested credential "
                        "or a clearly equivalent credential. "

                        "A course, introduction, training program, workshop, "
                        "or exposure to a subject is NOT proof that the "
                        "candidate holds a certification. "

                        "For example, 'Introduction to Lean Six Sigma' does "
                        "not prove 'Lean Six Sigma certification'. That should "
                        "normally be related_evidence rather than direct_match. "

                        "Confidence is confidence in your classification, "
                        "not the candidate's quality or match percentage. "

                        "Return confidence as a number from 0 to 100. "
                        "For example, use 95 for ninety-five percent, "
                        "not 0.95. "

                        "Evidence candidates are numbered starting at 1. "

                        "If evidence supports the requirement, put the number "
                        "of the strongest evidence candidate in "
                        "selected_evidence_index. "

                        "If no evidence supports the requirement, use null. "

                        "Do not invent, rename, or rewrite an evidence source. "

                        "Explain the decision briefly and clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"JOB REQUIREMENT\n"
                        f"{requirement.requirement}\n\n"

                        f"CATEGORY\n"
                        f"{requirement.category}\n\n"

                        f"IMPORTANCE\n"
                        f"{requirement.importance}\n\n"

                        f"RESUME EVIDENCE CANDIDATES\n"
                        f"{evidence_text}"
                    ),
                },
            ],
            text_format=ValidatedRequirement,
        )

    except Exception as exc:
        raise EvidenceValidationError(
            f"Evidence validation failed: {exc}"
        ) from exc

    result = response.output_parsed

    if result is None:
        raise EvidenceValidationError(
            "No validated requirement was returned."
        )

    # Never allow the LLM to invent evidence text/source.
    # Python copies them from the selected actual evidence item.
    if (
        result.selected_evidence_index is not None
        and 1
        <= result.selected_evidence_index
        <= len(evidence_candidates)
    ):
        selected = evidence_candidates[
            result.selected_evidence_index - 1
        ]

        result.evidence_text = selected.text
        result.evidence_source = selected.source

    else:
        result.evidence_text = None
        result.evidence_source = None

    return result
