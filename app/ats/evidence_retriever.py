from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.document_models import ResumeProfile


@dataclass
class EvidenceCandidate:
    source: str
    text: str


def build_resume_evidence(
    profile: ResumeProfile,
) -> list[EvidenceCandidate]:
    evidence: list[EvidenceCandidate] = []

    if profile.professional_summary:
        evidence.append(
            EvidenceCandidate(
                source="Professional Summary",
                text=profile.professional_summary,
            )
        )

    for skill in profile.skills:
        evidence.append(
            EvidenceCandidate(
                source="Skills",
                text=skill,
            )
        )

    for tool in profile.tools:
        evidence.append(
            EvidenceCandidate(
                source="Tools",
                text=tool,
            )
        )

    for domain in profile.domains:
        evidence.append(
            EvidenceCandidate(
                source="Domains",
                text=domain,
            )
        )

    for soft_skill in profile.soft_skills:
        evidence.append(
            EvidenceCandidate(
                source="Soft Skills",
                text=soft_skill,
            )
        )

    for experience in profile.experience:
        source = "Experience"

        if experience.company or experience.title:
            source = " — ".join(
                value
                for value in [
                    experience.company,
                    experience.title,
                ]
                if value
            )

        for responsibility in experience.responsibilities:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=responsibility,
                )
            )

        for achievement in experience.achievements:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=achievement,
                )
            )

        for tool in experience.tools:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=tool,
                )
            )

        for skill in experience.skills:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=skill,
                )
            )

    for project in profile.projects:
        source = f"Project: {project.name}"

        if project.description:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=project.description,
                )
            )

        for skill in project.skills:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=skill,
                )
            )

        for tool in project.tools:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=tool,
                )
            )

    for certification in profile.certifications:
        text = certification.name

        if certification.issuer:
            text += f" — {certification.issuer}"

        evidence.append(
            EvidenceCandidate(
                source="Certification",
                text=text,
            )
        )

    for education in profile.education:
        text = " — ".join(
            value
            for value in [
                education.degree,
                education.field_of_study,
                education.institution,
            ]
            if value
        )

        if text:
            evidence.append(
                EvidenceCandidate(
                    source="Education",
                    text=text,
                )
            )

    return evidence

CATEGORY_SOURCE_PREFERENCES = {
    "certification": {
        "Certification": 0.35,
    },
    "education": {
        "Education": 0.35,
    },
    "tool": {
        "Tools": 0.25,
    },
    "technical_skill": {
        "Skills": 0.25,
        "Tools": 0.10,
    },
    "soft_skill": {
        "Soft Skills": 0.10,
    },
    "domain": {
        "Domains": 0.25,
    },
}
def retrieve_best_evidence(
    requirement: str,
    evidence_records: list[EvidenceCandidate],
    category: str | None = None,
    top_k: int = 8,
) -> list[EvidenceCandidate]:

    if not evidence_records:
        return []

    texts = [
        requirement,
        *[record.text for record in evidence_records],
    ]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        vectors = vectorizer.fit_transform(texts)

    except ValueError:
        return evidence_records[:top_k]

    requirement_vector = vectors[0:1]
    evidence_vectors = vectors[1:]

    similarities = cosine_similarity(
        requirement_vector,
        evidence_vectors,
    )[0]

    if category:
        preferences = CATEGORY_SOURCE_PREFERENCES.get(
            category,
            {},
        )

        for index, record in enumerate(evidence_records):
            similarities[index] += preferences.get(
                record.source,
                0.0,
            )

            if category in {
                "soft_skill",
                "leadership",
                "responsibility",
                "business_impact",
            }:
                generic_sections = {
                    "Skills",
                    "Tools",
                    "Domains",
                    "Soft Skills",
                    "Certification",
                    "Education",
                    "Professional Summary",
                }

                if record.source not in generic_sections:
                    similarities[index] += 0.20

    ranked_indexes = similarities.argsort()[::-1]

    return [
        evidence_records[int(index)]
        for index in ranked_indexes[:top_k]
    ]
      
