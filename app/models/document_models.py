from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EducationRecord(BaseModel):
    degree: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    graduation_date: str | None = None


class CertificationRecord(BaseModel):
    name: str
    issuer: str | None = None
    date: str | None = None


class ExperienceRecord(BaseModel):
    company: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    responsibilities: list[str] = Field(
        default_factory=list
    )

    achievements: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )

    skills: list[str] = Field(
        default_factory=list
    )


class ProjectRecord(BaseModel):
    name: str
    description: str | None = None

    skills: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )


class ResumeProfile(BaseModel):
    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None

    professional_summary: str | None = None
    total_experience_years: float | None = None

    skills: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )

    domains: list[str] = Field(
        default_factory=list
    )

    soft_skills: list[str] = Field(
        default_factory=list
    )

    education: list[EducationRecord] = Field(
        default_factory=list
    )

    certifications: list[CertificationRecord] = Field(
        default_factory=list
    )

    experience: list[ExperienceRecord] = Field(
        default_factory=list
    )

    projects: list[ProjectRecord] = Field(
        default_factory=list
    )

    extraction_warnings: list[str] = Field(
        default_factory=list
    )


class JobRequirement(BaseModel):
    requirement: str

    category: Literal[
        "technical_skill",
        "tool",
        "certification",
        "soft_skill",
        "responsibility",
        "experience",
        "education",
        "domain",
        "methodology",
        "leadership",
        "business_impact",
        "other",
    ]

    importance: Literal[
        "required",
        "preferred",
        "unclear",
    ] = "unclear"


class JobProfile(BaseModel):
    job_title: str | None = None
    company: str | None = None
    summary: str | None = None

    responsibilities: list[str] = Field(
        default_factory=list
    )

    requirements: list[JobRequirement] = Field(
        default_factory=list
    )

    required_skills: list[str] = Field(
        default_factory=list
    )

    preferred_skills: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )

    domains: list[str] = Field(
        default_factory=list
    )

    soft_skills: list[str] = Field(
        default_factory=list
    )

    minimum_experience_years: float | None = None
    maximum_experience_years: float | None = None

    education_requirements: list[str] = Field(
        default_factory=list
    )

    certification_requirements: list[str] = Field(
        default_factory=list
    )

    extraction_warnings: list[str] = Field(
        default_factory=list
    )
