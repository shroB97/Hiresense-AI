from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.document_models import (
    JobProfile,
    JobRequirement,
    ResumeProfile,
)


MatchStatus = Literal[
    "direct_match",
    "related_evidence",
    "not_found",
    "uncertain",
]


class RequirementMatch(BaseModel):
    requirement: str
    category: str
    importance: str
    status: MatchStatus

    percentage_credit: float = Field(
        ge=0,
        le=100,
    )

    evidence: str | None = None
    evidence_source: str | None = None
    explanation: str


class MatchReport(BaseModel):
    overall_percentage: float = Field(
        ge=0,
        le=100,
    )

    skills_percentage: float = Field(
        ge=0,
        le=100,
    )

    experience_percentage: float = Field(
        ge=0,
        le=100,
    )

    responsibilities_percentage: float = Field(
        ge=0,
        le=100,
    )

    tools_percentage: float = Field(
        ge=0,
        le=100,
    )

    education_percentage: float = Field(
        ge=0,
        le=100,
    )

    certifications_percentage: float = Field(
        ge=0,
        le=100,
    )

    document_similarity_percentage: float = Field(
        ge=0,
        le=100,
    )

    matches: list[RequirementMatch]

    strengths: list[str]
    missing_requirements: list[str]
    uncertain_requirements: list[str]


CATEGORY_WEIGHTS = {
    "skill": 0.35,
    "experience": 0.25,
    "responsibility": 0.20,
    "tool": 0.10,
    "education": 0.05,
    "certification": 0.05,
}


STATUS_CREDIT = {
    "direct_match": 100.0,
    "related_evidence": 60.0,
    "uncertain": 25.0,
    "not_found": 0.0,
}


IMPORTANCE_MULTIPLIER = {
    "required": 1.0,
    "preferred": 0.45,
    "unclear": 0.70,
}


ALIASES = {
    "sap s/4hana": {
        "sap s/4hana",
        "s/4hana",
        "s4hana",
        "sap s4",
        "s/4 hana",
    },
    "power bi": {
        "power bi",
        "powerbi",
        "microsoft power bi",
    },
    "sap signavio": {
        "sap signavio",
        "signavio",
    },
    "jira": {
        "jira",
        "atlassian jira",
    },
    "stakeholder management": {
        "stakeholder management",
        "stakeholder engagement",
        "managed stakeholders",
        "worked with stakeholders",
    },
    "requirements elicitation": {
        "requirements elicitation",
        "requirements gathering",
        "captured user needs",
        "gathered requirements",
    },
}


def normalize(text: str | None) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9+#./ -]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def get_aliases(requirement: str) -> set[str]:
    normalized_requirement = normalize(requirement)
    matches = {normalized_requirement}

    for canonical, aliases in ALIASES.items():
        normalized_aliases = {
            normalize(alias)
            for alias in aliases
        }

        if (
            normalized_requirement == normalize(canonical)
            or normalized_requirement in normalized_aliases
        ):
            matches.update(normalized_aliases)
            matches.add(normalize(canonical))

    return matches


def build_resume_evidence(
    profile: ResumeProfile,
) -> list[tuple[str, str]]:
    evidence: list[tuple[str, str]] = []

    if profile.professional_summary:
        evidence.append(
            (
                "Professional summary",
                profile.professional_summary,
            )
        )

    for skill in profile.skills:
        evidence.append(("Skills", skill))

    for tool in profile.tools:
        evidence.append(("Tools", tool))

    for domain in profile.domains:
        evidence.append(("Domains", domain))

    for soft_skill in profile.soft_skills:
        evidence.append(("Soft skills", soft_skill))

    for experience in profile.experience:
        source = "Experience"

        if experience.company or experience.title:
            source = (
                f"Experience: "
                f"{experience.company or ''} "
                f"{experience.title or ''}"
            ).strip()

        for responsibility in experience.responsibilities:
            evidence.append((source, responsibility))

        for achievement in experience.achievements:
            evidence.append((source, achievement))

        for tool in experience.tools:
            evidence.append((source, tool))

        for skill in experience.skills:
            evidence.append((source, skill))

    for project in profile.projects:
        source = f"Project: {project.name}"

        if project.description:
            evidence.append(
                (source, project.description)
            )

        for skill in project.skills:
            evidence.append((source, skill))

        for tool in project.tools:
            evidence.append((source, tool))

    for education in profile.education:
        text = " ".join(
            value
            for value in [
                education.degree,
                education.field_of_study,
                education.institution,
            ]
            if value
        )

        if text:
            evidence.append(("Education", text))

    for certification in profile.certifications:
        evidence.append(
            (
                "Certification",
                certification.name,
            )
        )

    return evidence


def cosine_text_similarity(
    text_a: str,
    text_b: str,
) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        vectors = vectorizer.fit_transform(
            [text_a, text_b]
        )

        return float(
            cosine_similarity(
                vectors[0:1],
                vectors[1:2],
            )[0][0]
        )

    except ValueError:
        return 0.0


def direct_match(
    requirement: str,
    evidence_text: str,
) -> bool:
    normalized_evidence = normalize(evidence_text)

    return any(
        alias
        and alias in normalized_evidence
        for alias in get_aliases(requirement)
    )


def classify_requirement(
    requirement: JobRequirement,
    evidence_records: list[tuple[str, str]],
) -> RequirementMatch:
    for source, evidence in evidence_records:
        if direct_match(
            requirement.requirement,
            evidence,
        ):
            return RequirementMatch(
                requirement=requirement.requirement,
                category=requirement.category,
                importance=requirement.importance,
                status="direct_match",
                percentage_credit=100,
                evidence=evidence,
                evidence_source=source,
                explanation=(
                    "The requirement or a recognized equivalent "
                    "appears explicitly in the résumé."
                ),
            )

    best_source = None
    best_evidence = None
    best_similarity = 0.0

    for source, evidence in evidence_records:
        similarity = cosine_text_similarity(
            requirement.requirement,
            evidence,
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_source = source
            best_evidence = evidence

    if best_similarity >= 0.38:
        return RequirementMatch(
            requirement=requirement.requirement,
            category=requirement.category,
            importance=requirement.importance,
            status="related_evidence",
            percentage_credit=60,
            evidence=best_evidence,
            evidence_source=best_source,
            explanation=(
                "Related résumé evidence was found, but the exact "
                "requirement is not explicitly stated."
            ),
        )

    if best_similarity >= 0.20:
        return RequirementMatch(
            requirement=requirement.requirement,
            category=requirement.category,
            importance=requirement.importance,
            status="uncertain",
            percentage_credit=25,
            evidence=best_evidence,
            evidence_source=best_source,
            explanation=(
                "Possible related evidence was found, but it is "
                "not strong enough to confirm the requirement."
            ),
        )

    return RequirementMatch(
        requirement=requirement.requirement,
        category=requirement.category,
        importance=requirement.importance,
        status="not_found",
        percentage_credit=0,
        explanation=(
            "No sufficiently strong résumé evidence was found."
        ),
    )


def all_job_requirements(
    profile: JobProfile,
) -> list[JobRequirement]:
    requirements = list(profile.requirements)

    requirements.extend(
        JobRequirement(
            requirement=item,
            category="skill",
            importance="required",
        )
        for item in profile.required_skills
    )

    requirements.extend(
        JobRequirement(
            requirement=item,
            category="skill",
            importance="preferred",
        )
        for item in profile.preferred_skills
    )

    requirements.extend(
        JobRequirement(
            requirement=item,
            category="tool",
            importance="unclear",
        )
        for item in profile.tools
    )

    requirements.extend(
        JobRequirement(
            requirement=item,
            category="education",
            importance="required",
        )
        for item in profile.education_requirements
    )

    requirements.extend(
        JobRequirement(
            requirement=item,
            category="certification",
            importance="required",
        )
        for item in profile.certification_requirements
    )

    deduplicated = []
    seen = set()

    for requirement in requirements:
        key = (
            normalize(requirement.requirement),
            requirement.category,
        )

        if not key[0] or key in seen:
            continue

        seen.add(key)
        deduplicated.append(requirement)

    return deduplicated


def category_percentage(
    matches: list[RequirementMatch],
    category: str,
) -> float:
    category_matches = [
        match
        for match in matches
        if match.category == category
    ]

    if not category_matches:
        return 0.0

    total_weight = 0.0
    earned_weight = 0.0

    for match in category_matches:
        importance = IMPORTANCE_MULTIPLIER.get(
            match.importance,
            0.70,
        )

        total_weight += importance

        earned_weight += (
            importance
            * match.percentage_credit
        )

    return round(
        earned_weight / total_weight,
        1,
    )


def create_match_report(
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
) -> MatchReport:
    evidence = build_resume_evidence(
        resume_profile
    )

    requirements = all_job_requirements(
        job_profile
    )

    matches = [
        classify_requirement(
            requirement,
            evidence,
        )
        for requirement in requirements
    ]

    scores = {
        category: category_percentage(
            matches,
            category,
        )
        for category in CATEGORY_WEIGHTS
    }

    available_weight = sum(
        weight
        for category, weight in CATEGORY_WEIGHTS.items()
        if any(
            match.category == category
            for match in matches
        )
    )

    weighted_score = sum(
        scores[category] * weight
        for category, weight in CATEGORY_WEIGHTS.items()
        if any(
            match.category == category
            for match in matches
        )
    )

    overall = (
        weighted_score / available_weight
        if available_weight
        else 0.0
    )

    resume_search_text = "\n".join(
        text
        for _, text in evidence
    )

    job_search_text = "\n".join(
        requirement.requirement
        for requirement in requirements
    )

    document_similarity = (
        cosine_text_similarity(
            resume_search_text,
            job_search_text,
        )
        * 100
    )

    return MatchReport(
        overall_percentage=round(overall, 1),
        skills_percentage=scores["skill"],
        experience_percentage=scores["experience"],
        responsibilities_percentage=scores[
            "responsibility"
        ],
        tools_percentage=scores["tool"],
        education_percentage=scores["education"],
        certifications_percentage=scores[
            "certification"
        ],
        document_similarity_percentage=round(
            document_similarity,
            1,
        ),
        matches=matches,
        strengths=[
            match.requirement
            for match in matches
            if match.status == "direct_match"
        ],
        missing_requirements=[
            match.requirement
            for match in matches
            if match.status == "not_found"
        ],
        uncertain_requirements=[
            match.requirement
            for match in matches
            if match.status == "uncertain"
        ],
    )
