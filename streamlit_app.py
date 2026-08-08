from __future__ import annotations
from app.ats.matching_engine import (
    MatchReport,
    create_match_report,
)

from app.extractors.structured_extractor import (
    StructuredExtractionError,
    extract_job_profile,
    extract_resume_profile,
)
from app.recommendations.resume_optimizer import (
    ResumeOptimizationError,
    ResumeOptimizationReport,
    optimize_resume,
)
import pandas as pd
import plotly.express as px
import streamlit as st
from app.parsers.resume_parser import (
    ResumeParserError,
    parse_resume,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HireSense AI",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM STYLES
# =========================================================

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap'
        );

        html,
        body,
        [data-testid="stAppViewContainer"],
        button,
        input,
        textarea,
        label,
        p,
        span {
            font-family: "Manrope", sans-serif;
        }

        .stApp {
            background: #ffffff;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.95);
            border-bottom: 1px solid #f3f4f6;
            backdrop-filter: blur(12px);
        }

        [data-testid="stToolbar"] {
            visibility: hidden;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }

        h1,
        h2,
        h3 {
            color: #111827;
            letter-spacing: -0.035em;
        }

        p {
            color: #4b5563;
        }

        .hs-brand {
            color: #111827;
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }

        .hs-brand span {
            color: #c2410c;
        }

        .hs-kicker {
            color: #c2410c;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        /* Hero panel */

        .st-key-hero_panel {
            background:
                radial-gradient(
                    circle at 88% 15%,
                    rgba(234, 88, 12, 0.13),
                    transparent 32%
                ),
                linear-gradient(
                    135deg,
                    #ffffff,
                    #fff7ed
                );

            border: 1px solid #fed7aa !important;
            border-radius: 28px;
            padding: 2.5rem;
            margin-bottom: 2.5rem;

            box-shadow:
                0 22px 60px rgba(194, 65, 12, 0.08);
        }

        .st-key-hero_panel h1 {
            max-width: 850px;
            font-size: clamp(3rem, 6vw, 5.3rem);
            line-height: 1.04;
            font-weight: 800;
            letter-spacing: -0.055em;
        }

        .st-key-hero_panel p {
            max-width: 820px;
            font-size: 1.08rem;
            line-height: 1.75;
        }

        /* Input panel */

        .st-key-input_panel {
            border: 1px solid #fed7aa !important;
            border-radius: 20px;
            background: #ffffff;
            padding: 1.25rem;
            box-shadow:
                0 10px 30px rgba(17, 24, 39, 0.05);
        }

        /* File uploader */

        div[data-testid="stFileUploader"] {
            min-height: 275px;
            padding: 0.8rem;
            border: 1px solid #fed7aa;
            border-radius: 18px;
            background: #ffffff;
        }

        div[data-testid="stFileUploader"] section {
            min-height: 175px;
            border: 1px dashed #ea580c;
            border-radius: 14px;
            background: #fff7ed;
        }

        div[data-testid="stFileUploader"] button {
            color: #c2410c;
            background: #ffffff;
            border: 1px solid #fed7aa;
            border-radius: 10px;
            font-weight: 700;
        }

        /* Text area */

        div[data-testid="stTextArea"] textarea {
            min-height: 250px;
            padding: 1rem;

            color: #111827;
            background: #ffffff;

            border: 1px solid #fed7aa;
            border-radius: 16px;

            line-height: 1.55;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #c2410c;
            box-shadow:
                0 0 0 2px rgba(194, 65, 12, 0.12);
        }

        /* Primary button */

        .stButton > button {
            min-height: 3.2rem;
            width: 100%;

            color: #ffffff;
            background:
                linear-gradient(
                    90deg,
                    #c2410c,
                    #ea580c
                );

            border: none;
            border-radius: 14px;

            font-size: 1rem;
            font-weight: 800;

            box-shadow:
                0 12px 26px rgba(194, 65, 12, 0.23);
        }

        .stButton > button:hover {
            color: #ffffff;
            background: #9a3412;
            border: none;
        }

        /* Summary cards */

        .st-key-resume_summary,
        .st-key-jd_summary {
            border: 1px solid #fed7aa !important;
            border-radius: 16px;
            background: #fff7ed;
            min-height: 105px;
        }

        .st-key-resume_summary p,
        .st-key-jd_summary p {
            color: #111827;
        }

        /* Workflow and feature cards */

        .st-key-workflow_1,
        .st-key-workflow_2,
        .st-key-workflow_3,
        .st-key-workflow_4,
        .st-key-feature_1,
        .st-key-feature_2,
        .st-key-feature_3 {
            border: 1px solid #e5e7eb !important;
            border-radius: 18px;
            background: #ffffff;

            box-shadow:
                0 8px 25px rgba(17, 24, 39, 0.045);
        }

        .st-key-workflow_1,
        .st-key-workflow_2,
        .st-key-workflow_3,
        .st-key-workflow_4 {
            min-height: 220px;
        }

        .st-key-feature_1,
        .st-key-feature_2,
        .st-key-feature_3 {
            min-height: 285px;
            border-top: 4px solid #c2410c !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        hr {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin-top: 3rem;
            margin-bottom: 3rem;
        }

        @media (max-width: 800px) {
            .st-key-hero_panel {
                padding: 1.5rem;
            }
        }
	.match-score-panel {
    padding: 1.6rem;
    border: 1px solid #fed7aa;
    border-radius: 20px;
    background:
        linear-gradient(
            135deg,
            #ffffff,
            #fff7ed
        );
    text-align: center;
    box-shadow:
        0 12px 30px rgba(194, 65, 12, 0.08);
}

.match-score-number {
    color: #c2410c;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
}

.match-score-label {
    margin-top: 0.5rem;
    color: #6b7280;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.score-excellent {
    color: #15803d;
    font-weight: 700;
}

.score-strong {
    color: #c2410c;
    font-weight: 700;
}

.score-moderate {
    color: #b45309;
    font-weight: 700;
}

.score-low {
    color: #b91c1c;
    font-weight: 700;
}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SMALL DECORATIVE LABEL
# =========================================================

def render_kicker(text: str) -> None:
    st.markdown(
        f'<div class="hs-kicker">{text}</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# HEADER
# =========================================================

def render_header() -> None:
    st.markdown(
        '<div class="hs-brand">HireSense <span>AI</span></div>',
        unsafe_allow_html=True,
    )


# =========================================================
# HERO
# =========================================================

def render_hero() -> None:
    with st.container(
        border=True,
        key="hero_panel",
    ):
        render_kicker(
            "AI Resume Intelligence & Interview Copilot"
        )

        st.title(
            "Make every application more intentional."
        )

        st.write(
            "Upload your resume and paste a job description to receive "
            "an evidence-based ATS analysis, requirement matching, "
            "missing-skill guidance, resume recommendations, and a "
            "personalized interview preparation roadmap."
        )

        trust_col1, trust_col2, trust_col3 = st.columns(3)

        with trust_col1:
            st.write("✓ Evidence-based matching")

        with trust_col2:
            st.write("✓ Transparent scoring")

        with trust_col3:
            st.write("✓ Personalized preparation")


# =========================================================
# INPUT SECTION
# =========================================================

def render_input_section():
    render_kicker("Start your analysis")

    st.header(
        "Create your personalized career report"
    )

    st.caption(
        "Provide your resume and the complete job description. "
        "HireSense AI will compare both inputs and prepare a "
        "structured fit assessment."
    )

    with st.container(
        border=True,
        key="input_panel",
    ):
        left_column, right_column = st.columns(
            2,
            gap="large",
        )

        with left_column:
            st.subheader(
                "1. Upload your resume"
            )

            uploaded_resume = st.file_uploader(
                "Supported formats: PDF and DOCX",
                type=["pdf", "docx"],
                help=(
                    "The resume is processed only for the "
                    "current analysis."
                ),
            )

        with right_column:
            st.subheader(
                "2. Paste the job description"
            )

            job_description = st.text_area(
                (
                    "Include responsibilities, requirements, tools, "
                    "education, and preferred qualifications."
                ),
                height=270,
                placeholder=(
                    "Paste the complete job description here...\n\n"
                    "What You Will Do...\n\n"
                    "What We Are Looking For..."
                ),
            )

    return uploaded_resume, job_description


# =========================================================
# INPUT VALIDATION
# =========================================================
def get_match_label(
    percentage: float,
) -> tuple[str, str]:
    if percentage >= 85:
        return (
            "Excellent alignment",
            "score-excellent",
        )

    if percentage >= 70:
        return (
            "Strong alignment",
            "score-strong",
        )

    if percentage >= 50:
        return (
            "Moderate alignment",
            "score-moderate",
        )

    return (
        "Low alignment",
        "score-low",
    )


def render_score_card(score: float):

    match_label, _ = get_match_label(score)

    with st.container(border=True):

        st.markdown(
            f"""
            <h1 style="
                text-align:center;
                color:#c2410c;
                font-size:72px;
                margin-bottom:0px;
            ">
                {score:.0f}%
            </h1>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h4 style='text-align:center; color:#6b7280;'>"
            "HireSense Match Score"
            "</h4>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<h3 style='text-align:center; color:#ea580c;'>"
            f"{match_label}"
            f"</h3>",
            unsafe_allow_html=True,
        )


def render_category_scores(
    report: MatchReport,
) -> None:
    st.markdown("### Match by Category")

    category_scores = [
        (
            "Skills",
            report.skills_percentage,
        ),
        (
            "Experience",
            report.experience_percentage,
        ),
        (
            "Responsibilities",
            report.responsibilities_percentage,
        ),
        (
            "Tools",
            report.tools_percentage,
        ),
        (
            "Education",
            report.education_percentage,
        ),
        (
            "Certifications",
            report.certifications_percentage,
        ),
    ]

    first_row = st.columns(3)
    second_row = st.columns(3)

    for index, (
        category,
        percentage,
    ) in enumerate(category_scores):
        target_column = (
            first_row[index]
            if index < 3
            else second_row[index - 3]
        )

        with target_column:
            st.metric(
                category,
                f"{percentage:.0f}%",
            )

            st.progress(
                int(
                    max(
                        0,
                        min(
                            100,
                            percentage,
                        ),
                    )
                )
            )

def render_match_chart(
    report: MatchReport,
) -> None:
    st.markdown("### Match Breakdown")

    chart_data = pd.DataFrame(
        {
            "Category": [
                "Skills",
                "Experience",
                "Responsibilities",
                "Tools",
                "Education",
                "Certifications",
            ],
            "Match": [
                report.skills_percentage,
                report.experience_percentage,
                report.responsibilities_percentage,
                report.tools_percentage,
                report.education_percentage,
                report.certifications_percentage,
            ],
        }
    )

    fig = px.bar(
        chart_data,
        x="Match",
        y="Category",
        orientation="h",
        text="Match",
        range_x=[0, 100],
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="outside",
    )

    fig.update_layout(
        height=420,
        margin=dict(
            l=20,
            r=40,
            t=20,
            b=20,
        ),
        xaxis_title="Match Percentage",
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

def render_requirement_summary(
    report: MatchReport,
) -> None:
    direct_matches = sum(
        match.status == "direct_match"
        for match in report.matches
    )

    related_matches = sum(
        match.status == "related_evidence"
        for match in report.matches
    )

    uncertain_matches = sum(
        match.status == "uncertain"
        for match in report.matches
    )

    missing_matches = sum(
        match.status == "not_found"
        for match in report.matches
    )

    st.markdown("### Requirement Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Direct Matches",
            direct_matches,
        )

    with col2:
        st.metric(
            "Related Evidence",
            related_matches,
        )

    with col3:
        st.metric(
            "Uncertain",
            uncertain_matches,
        )

    with col4:
        st.metric(
            "Not Found",
            missing_matches,
        )


def build_requirement_dataframe(
    report: MatchReport,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    status_labels = {
        "direct_match": "Direct Match",
        "related_evidence": "Related Evidence",
        "uncertain": "Uncertain",
        "not_found": "Not Found",
    }

    for match in report.matches:
        rows.append(
            {
                "Requirement": match.requirement,
                "Category": (
                    match.category
                    .replace("_", " ")
                    .title()
                ),
                "Importance": (
                    match.importance.title()
                ),
                "Status": status_labels[
                    match.status
                ],
                "Match Credit": (
                    f"{match.percentage_credit:.0f}%"
                ),
                "Evidence Source": (
                    match.evidence_source
                    or "Not available"
                ),
            }
        )

    return pd.DataFrame(rows)


def render_gap_analysis(
    report: MatchReport,
) -> None:
    st.markdown("### Gap Analysis")

    matched_column, gap_column = st.columns(
        2,
        gap="large",
    )

    with matched_column:
        with st.container(border=True):
            st.markdown(
                "#### Strong Alignments"
            )

            if report.strengths:
                for strength in report.strengths:
                    st.write(f"✓ {strength}")
            else:
                st.write(
                    "No direct alignments were identified."
                )

    with gap_column:
        with st.container(border=True):
            st.markdown(
                "#### Missing Requirements"
            )

            if report.missing_requirements:
                for requirement in (
                    report.missing_requirements
                ):
                    st.write(f"✕ {requirement}")
            else:
                st.write(
                    "No missing requirements were identified."
                )

    if report.uncertain_requirements:
        with st.expander(
            "Review uncertain requirements"
        ):
            for requirement in (
                report.uncertain_requirements
            ):
                st.write(f"• {requirement}")


def render_requirement_evidence(
    report: MatchReport,
) -> None:
    st.markdown("### Requirement Evidence")

    requirement_table = (
        build_requirement_dataframe(
            report
        )
    )

    status_filter = st.multiselect(
        "Filter by match status",
        options=[
            "Direct Match",
            "Related Evidence",
            "Uncertain",
            "Not Found",
        ],
        default=[
            "Direct Match",
            "Related Evidence",
            "Uncertain",
            "Not Found",
        ],
    )

    if status_filter:
        filtered_table = requirement_table[
            requirement_table[
                "Status"
            ].isin(status_filter)
        ]
    else:
        filtered_table = requirement_table

    st.dataframe(
        filtered_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### Detailed Evidence"
    )

    status_labels = {
        "direct_match": "Direct Match",
        "related_evidence": "Related Evidence",
        "uncertain": "Uncertain",
        "not_found": "Not Found",
    }

    for index, match in enumerate(
        report.matches,
        start=1,
    ):
        status_label = status_labels[
            match.status
        ]

        with st.expander(
            f"{index}. {match.requirement} "
            f"— {status_label} "
            f"({match.percentage_credit:.0f}%)"
        ):
            detail_col1, detail_col2 = (
                st.columns(2)
            )

            with detail_col1:
                st.write(
                    f"**Category:** "
                    f"{match.category.replace('_', ' ').title()}"
                )

                st.write(
                    f"**Importance:** "
                    f"{match.importance.title()}"
                )

            with detail_col2:
                st.write(
                    f"**Match credit:** "
                    f"{match.percentage_credit:.0f}%"
                )

                st.write(
                    f"**Evidence source:** "
                    f"{match.evidence_source or 'Not available'}"
                )

            st.write(
                f"**Assessment:** "
                f"{match.explanation}"
            )

            if match.evidence:
                st.write(
                    "**Résumé evidence:**"
                )

                st.info(
                    match.evidence
                )


def render_match_dashboard(
    report: MatchReport,
) -> None:
    st.divider()

    render_kicker(
        "Evidence-based fit assessment"
    )

    st.header(
        "HireSense Match Report"
    )

    score_column, summary_column = (
        st.columns(
            [1, 2],
            gap="large",
        )
    )

    with score_column:
        render_score_card(
            report.overall_percentage
        )

    with summary_column:
        st.markdown(
            "### Overall résumé-to-job alignment"
        )

        st.progress(
            int(
                max(
                    0,
                    min(
                        100,
                        report.overall_percentage,
                    ),
                )
            )
        )

        st.write(
            "The score combines skills, experience, "
            "responsibilities, tools, education, and "
            "certification alignment."
        )

        st.metric(
            "Document Similarity",
            (
                f"{report.document_similarity_percentage:.0f}%"
            ),
        )

    render_category_scores(report)
    render_match_chart(report)

    st.divider()

    render_requirement_summary(report)

    st.divider()

    render_gap_analysis(report)

    st.divider()

    render_requirement_evidence(report)

    st.caption(
        "The HireSense Match Score is an evidence-based "
        "résumé-to-job alignment estimate. It is not an "
        "employer-issued ATS score or hiring decision."
    )
def render_resume_optimizer(
    report: ResumeOptimizationReport,
) -> None:

    st.divider()

    render_kicker(
        "AI resume optimization"
    )

    st.header(
        "Resume Improvement Recommendations"
    )

    st.caption(
        "Suggestions are restricted to experience already "
        "supported by your resume."
    )

    if report.professional_summary_suggestion:
        with st.container(border=True):
            st.markdown(
                "### Suggested Professional Summary"
            )

            st.write(
                report.professional_summary_suggestion
            )

    if report.bullet_suggestions:
        st.markdown(
            "### Suggested Bullet Improvements"
        )

        for index, suggestion in enumerate(
            report.bullet_suggestions,
            start=1,
        ):
            with st.expander(
                f"{index}. "
                f"{suggestion.target_requirement}"
            ):
                st.write(
                    "**Resume evidence:**"
                )

                st.info(
                    suggestion.original_evidence
                )

                st.write(
                    "**Suggested rewrite:**"
                )

                st.success(
                    suggestion.suggested_rewrite
                )

                st.write(
                    f"**Why:** {suggestion.reason}"
                )

    if report.keywords_to_emphasize:
        st.markdown(
            "### Keywords Already Supported by Your Resume"
        )

        st.write(
            " · ".join(
                report.keywords_to_emphasize
            )
        )

    if report.missing_but_unverified_skills:
        st.markdown(
            "### Requirements Not Verified in Your Resume"
        )

        st.warning(
            "Do not add these unless you genuinely "
            "have this experience."
        )

        for skill in (
            report.missing_but_unverified_skills
        ):
            st.write(
                f"• {skill}"
            )

    if report.warnings:
        with st.expander(
            "Optimization warnings"
        ):
            for warning in report.warnings:
                st.write(
                    f"• {warning}"
                )
def render_resume_intelligence(
    resume_profile,
) -> None:
    st.divider()

    render_kicker(
        "Resume intelligence"
    )

    st.header(
        "Candidate Profile"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Candidate",
            resume_profile.candidate_name
            or "Not detected",
        )

    with col2:
        if (
            resume_profile.total_experience_years
            is not None
        ):
            experience_value = (
                f"{resume_profile.total_experience_years:.1f} years"
            )
        else:
            experience_value = (
                "Not determined"
            )

        st.metric(
            "Experience",
            experience_value,
        )

    with col3:
        st.metric(
            "Skills Detected",
            len(resume_profile.skills),
        )

    if resume_profile.professional_summary:
        with st.container(border=True):
            st.markdown(
                "### Professional Summary"
            )

            st.write(
                resume_profile.professional_summary
            )

    if resume_profile.skills:
        st.markdown(
            "### Skills"
        )

        st.write(
            " · ".join(
                resume_profile.skills[:20]
            )
        )

    if resume_profile.tools:
        st.markdown(
            "### Tools & Technologies"
        )

        st.write(
            " · ".join(
                resume_profile.tools[:20]
            )
        )

    if resume_profile.domains:
        st.markdown(
            "### Domain Experience"
        )

        st.write(
            " · ".join(
                resume_profile.domains[:15]
            )
        )

    if resume_profile.education:
        st.markdown(
            "### Education"
        )

        for education in (
            resume_profile.education
        ):
            education_parts = [
                education.degree,
                education.field_of_study,
                education.institution,
            ]

            education_text = " — ".join(
                value
                for value in education_parts
                if value
            )

            if education_text:
                st.write(
                    f"• {education_text}"
                )

    if resume_profile.certifications:
        st.markdown(
            "### Certifications"
        )

        for certification in (
            resume_profile.certifications
        ):
            st.write(
                f"• {certification.name}"
            )
def render_job_intelligence(
    job_profile,
) -> None:
    st.divider()

    render_kicker(
        "Job intelligence"
    )

    st.header(
        "Role Requirements"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Job Title",
            job_profile.job_title
            or "Not provided",
        )

    with col2:
        if (
            job_profile.minimum_experience_years
            is not None
        ):
            experience_value = (
                f"{job_profile.minimum_experience_years:.0f}+ years"
            )
        else:
            experience_value = (
                "Not specified"
            )

        st.metric(
            "Minimum Experience",
            experience_value,
        )

    with col3:
        st.metric(
            "Requirements",
            len(job_profile.requirements),
        )

    if job_profile.summary:
        with st.container(border=True):
            st.markdown(
                "### Role Summary"
            )

            st.write(
                job_profile.summary
            )

    if job_profile.required_skills:
        st.markdown(
            "### Required Skills"
        )

        st.write(
            " · ".join(
                job_profile.required_skills[:20]
            )
        )

    if job_profile.preferred_skills:
        st.markdown(
            "### Preferred Skills"
        )

        st.write(
            " · ".join(
                job_profile.preferred_skills[:20]
            )
        )

    if job_profile.tools:
        st.markdown(
            "### Tools & Technologies"
        )

        st.write(
            " · ".join(
                job_profile.tools[:20]
            )
        )

    if job_profile.responsibilities:
        st.markdown(
            "### Key Responsibilities"
        )

        for responsibility in (
            job_profile.responsibilities[:10]
        ):
            st.write(
                f"• {responsibility}"
            )
def render_statistics() -> None:
    st.markdown("### Why HireSense AI?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Resume Formats", "PDF/DOCX")

    with col2:
        st.metric("Analysis Layers", "5")

    with col3:
        st.metric("Evidence Based", "Yes")

    with col4:
        st.metric("Explainable Matching", "100%")

def render_analysis_result(
    uploaded_resume,
    job_description: str,
) -> None:

    if uploaded_resume is None:
        st.warning(
            "Please upload a PDF or DOCX resume."
        )
        return

    if not job_description.strip():
        st.warning(
            "Please paste the complete job description."
        )
        return

    progress = st.progress(0)
    status = st.empty()

    try:

        status.write("Reading resume...")

        resume_result = parse_resume(
            filename=uploaded_resume.name,
            file_bytes=uploaded_resume.getvalue(),
        )

        progress.progress(20)

        status.write("Understanding resume...")

        resume_profile = extract_resume_profile(
            resume_result.text
        )

        progress.progress(45)

        status.write("Understanding job description...")

        job_profile = extract_job_profile(
            job_description
        )

        progress.progress(65)

        status.write("Comparing resume with job requirements...")

        match_report = create_match_report(
            resume_profile,
            job_profile,
        )

        progress.progress(82)

        status.write("Generating resume recommendations...")

        optimization_report = optimize_resume(
            resume_profile=resume_profile,
            job_profile=job_profile,
        )

        progress.progress(100)

        status.success("Career report ready.")
            

    except ResumeParserError as exc:
        st.error(str(exc))
        return

    except StructuredExtractionError as exc:
        st.error(str(exc))
        return

    except ResumeOptimizationError as exc:
        st.error(str(exc))
        return

    except Exception as exc:
        st.error(
            "An unexpected error occurred during analysis."
        )

        with st.expander("Technical details"):
            st.code(str(exc))

        return

    st.success(
        "Resume and job description analyzed successfully."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "File Type",
            resume_result.file_type,
        )

    with metric_col2:
        st.metric(
            "Resume Words",
            f"{resume_result.word_count:,}",
        )

    with metric_col3:
        page_value = (
            resume_result.page_count
            if resume_result.page_count is not None
            else "Not available"
        )

        st.metric(
            "Pages",
            page_value,
        )

    summary_col1, summary_col2 = st.columns(
        2,
        gap="large",
    )

    with summary_col1:
        with st.container(
            border=True,
            key="resume_summary",
        ):
            st.caption("RESUME RECEIVED")

            st.write(
                f"**{resume_result.filename}**"
            )

    with summary_col2:
        with st.container(
            border=True,
            key="jd_summary",
        ):
            st.caption(
                "JOB DESCRIPTION RECEIVED"
            )

            st.write(
                f"**{len(job_description):,} characters**"
            )

    render_resume_intelligence(
        resume_profile
    )

    render_job_intelligence(
        job_profile
    )

    render_match_dashboard(
        match_report
    )

    render_resume_optimizer(
        optimization_report
    )

    st.divider()

    st.subheader(
        "Extracted Resume Preview"
    )

    preview_length = 4_000

    preview_text = resume_result.text[
        :preview_length
    ]

    st.text_area(
        "Resume text",
        value=preview_text,
        height=350,
        disabled=True,
    )

    if len(resume_result.text) > preview_length:
        st.caption(
            "Preview limited to the first "
            f"{preview_length:,} characters."
        )

    with st.expander(
        "View complete extracted resume text"
    ):
        st.text(
            resume_result.text
        )

# =========================================================
# WORKFLOW
# =========================================================

def render_workflow() -> None:
    render_kicker("Workflow")

    st.header(
        "How HireSense AI works"
    )

    st.caption(
        "A guided journey from uploaded documents to "
        "evidence-based career intelligence and interview preparation."
    )

    workflow_steps = [
        (
            "Step 01",
            "Resume Intelligence",
            (
                "Extract skills, experience, education, "
                "certifications, projects, and measurable achievements."
            ),
        ),
        (
            "Step 02",
            "Requirement Analysis",
            (
                "Identify responsibilities, required tools, "
                "experience, education, and preferred qualifications."
            ),
        ),
        (
            "Step 03",
            "Evidence Matching",
            (
                "Compare every job requirement with supporting "
                "evidence found in the resume."
            ),
        ),
        (
            "Step 04",
            "Interview Readiness",
            (
                "Generate likely interview stages, role-specific "
                "questions, and a personalized learning roadmap."
            ),
        ),
    ]

    workflow_columns = st.columns(
        4,
        gap="medium",
    )

    for index, workflow_step in enumerate(
        workflow_steps,
        start=1,
    ):
        step_number, title, description = workflow_step

        with workflow_columns[index - 1]:
            with st.container(
                border=True,
                key=f"workflow_{index}",
            ):
                st.caption(
                    step_number.upper()
                )

                st.subheader(title)

                st.write(description)


# =========================================================
# FEATURE CARDS
# =========================================================

def render_features() -> None:
    render_kicker("Career report")

    st.header(
        "What your analysis will include"
    )

    st.caption(
        "HireSense AI goes beyond a basic ATS percentage "
        "by explaining how the result was calculated and "
        "what to do next."
    )

    features = [
        (
            "ATS Intelligence",
            [
                "Overall match score",
                "Skills and tools alignment",
                "Keyword coverage",
                "Experience and education match",
                "Evidence for each requirement",
            ],
        ),
        (
            "Resume Optimization",
            [
                "Missing requirement explanations",
                "Recommended keywords",
                "Suggested resume bullets",
                "Tailored professional summary",
                "Role-specific priorities",
            ],
        ),
        (
            "Interview Readiness",
            [
                "Likely hiring stages",
                "Recruiter-screen questions",
                "Hiring-manager questions",
                "Technical preparation topics",
                "Personalized learning roadmap",
            ],
        ),
    ]

    feature_columns = st.columns(
        3,
        gap="large",
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):
        title, items = feature

        with feature_columns[index - 1]:
            with st.container(
                border=True,
                key=f"feature_{index}",
            ):
                st.subheader(title)

                for item in items:
                    st.write(
                        f"• {item}"
                    )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main() -> None:
    inject_styles()
    render_header()
    render_hero()

    render_statistics()

    uploaded_resume, job_description = (
        render_input_section()
    )

    st.write("")

    analyze_clicked = st.button(
        "Generate Career Report",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        render_analysis_result(
            uploaded_resume=uploaded_resume,
            job_description=job_description,
        )

    st.divider()

    render_workflow()

    st.divider()

    render_features()

    st.divider()

    st.caption(
        "HireSense AI · Evidence-based resume intelligence, "
        "ATS optimization, and interview preparation"
    )


if __name__ == "__main__":
    main()
