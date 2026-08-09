from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ats.evidence_validator import (
    EvidenceCandidate,
)
from app.models.document_models import (
    ResumeProfile,
)


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

    for record in profile.experience:

        source = "Experience"

        if record.company or record.title:
            source = " — ".join(
                item
                for item in [
                    record.company,
                    record.title,
                ]
                if item
            )

        for responsibility in record.responsibilities:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=responsibility,
                )
            )

        for achievement in record.achievements:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=achievement,
                )
            )

        for tool in record.tools:
            evidence.append(
                EvidenceCandidate(
                    source=source,
                    text=tool,
                )
            )

        for skill in record.skills:
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

        certification_text = certification.name

        if certification.issuer:
            certification_text += (
                f" — {certification.issuer}"
            )

        evidence.append(
            EvidenceCandidate(
                source="Certification",
                text=certification_text,
            )
        )

    for education in profile.education:

        education_text = " — ".join(
            value
            for value in [
                education.degree,
                education.field_of_study,
                education.institution,
            ]
            if value
        )

        if education_text:
            evidence.append(
                EvidenceCandidate(
                    source="Education",
                    text=education_text,
                )
            )

    return evidence


def retrieve_best_evidence(
    requirement: str,
    evidence_records: list[EvidenceCandidate],
    top_k: int = 5,
) -> list[EvidenceCandidate]:

    if not evidence_records:
        return []

    texts = [
        requirement,
        *[
            record.text
            for record in evidence_records
        ],
    ]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        vectors = vectorizer.fit_transform(
            texts
        )

    except ValueError:
        return evidence_records[:top_k]

    requirement_vector = vectors[0:1]
    evidence_vectors = vectors[1:]

    similarities = cosine_similarity(
        requirement_vector,
        evidence_vectors,
    )[0]

    ranked_indexes = similarities.argsort()[
        ::-1
    ]

    selected: list[EvidenceCandidate] = []

    for index in ranked_indexes[:top_k]:
        selected.append(
            evidence_records[int(index)]
        )

    return selected
