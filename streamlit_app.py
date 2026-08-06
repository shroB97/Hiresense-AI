from __future__ import annotations

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

    try:
        resume_result = parse_resume(
            filename=uploaded_resume.name,
            file_bytes=uploaded_resume.getvalue(),
        )

    except ResumeParserError as exc:
        st.error(str(exc))
        return

    except Exception as exc:
        st.error(
            "An unexpected error occurred while "
            "reading the resume."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(str(exc))

        return

    st.success(
        "Resume text extracted successfully."
    )

    metric_col1, metric_col2, metric_col3 = (
        st.columns(3)
    )

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
            if resume_result.page_count
            is not None
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
                f"**{len(job_description):,} "
                "characters**"
            )

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

    st.info(
        "The resume is now ready for skill extraction "
        "and ATS requirement matching."
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
