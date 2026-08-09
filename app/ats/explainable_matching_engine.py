from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from app.ats.evidence_retriever import (
    build_resume_evidence,
    retrieve_best_evidence,
)
from app.ats.evidence_validator import (
    ValidatedRequirement,
    validate_requirement,
)
from app.models.document_models import (
    JobProfile,
    ResumeProfile,
)


class CategoryResult(BaseModel):
    category: str
    score: float = Field(
        ge=0,
        le=100,
    )
    requirements: int


class ExplainableMatchReport(BaseModel):
    overall_score: float = Field(
        ge=0,
        le=100,
    )

    requirement_results: list[
        ValidatedRequirement
    ]

    category_results: list[
        CategoryResult
    ]

    direct_matches: int
    related_matches: int
    uncertain_matches: int
    missing_matches: int


STATUS_SCORE = {
    "direct_match": 100.0,
    "related_evidence": 65.0,
    "uncertain": 25.0,
    "not_found": 0.0,
}


IMPORTANCE_WEIGHT = {
    "required": 1.0,
    "preferred": 0.50,
    "unclear": 0.70,
}


def create_explainable_match_report(
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
) -> ExplainableMatchReport:

    resume_evidence = build_resume_evidence(
        resume_profile
    )

    results: list[
        ValidatedRequirement
    ] = []

    for requirement in job_profile.requirements:

        candidates = retrieve_best_evidence(
            requirement=requirement.requirement,
            evidence_records=resume_evidence,
            category=requirement.category,
            top_k=8,
         )

        result = validate_requirement(
            requirement=requirement,
            evidence_candidates=candidates,
        )

        results.append(result)

    total_weight = 0.0
    weighted_points = 0.0

    category_points = defaultdict(float)
    category_weights = defaultdict(float)
    category_counts = defaultdict(int)

    for result in results:

        weight = IMPORTANCE_WEIGHT.get(
            result.importance,
            0.70,
        )

        score = STATUS_SCORE[
            result.status
        ]

        total_weight += weight
        weighted_points += score * weight

        category_points[
            result.category
        ] += score * weight

        category_weights[
            result.category
        ] += weight

        category_counts[
            result.category
        ] += 1

    overall_score = (
        weighted_points / total_weight
        if total_weight
        else 0.0
    )

    category_results: list[
        CategoryResult
    ] = []

    for category in category_counts:

        category_score = (
            category_points[category]
            / category_weights[category]
            if category_weights[category]
            else 0.0
        )

        category_results.append(
            CategoryResult(
                category=category,
                score=round(
                    category_score,
                    1,
                ),
                requirements=(
                    category_counts[
                        category
                    ]
                ),
            )
        )

    return ExplainableMatchReport(
        overall_score=round(
            overall_score,
            1,
        ),
        requirement_results=results,
        category_results=category_results,
        direct_matches=sum(
            result.status == "direct_match"
            for result in results
        ),
        related_matches=sum(
            result.status
            == "related_evidence"
            for result in results
        ),
        uncertain_matches=sum(
            result.status == "uncertain"
            for result in results
        ),
        missing_matches=sum(
            result.status == "not_found"
            for result in results
        ),
    )
