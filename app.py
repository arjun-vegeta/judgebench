"""
JudgeBench Streamlit Interactive Demo Application
Benchmarking LLM-as-a-Judge (Gemini 3.1 Flash Lite & Groq Llama 3.3) vs. Classical ML Judge & Human Gold Set.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="JudgeBench — LLM Evaluation Harness",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .judge-card {
        border-radius: 12px;
        padding: 20px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Imports from project modules
from judges.llm_judge_a import LLMJudgeA  # noqa: E402
from judges.llm_judge_b import LLMJudgeB  # noqa: E402
from judges.ml_judge import ClassicalMLJudge  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_CSV = os.path.join(DATA_DIR, "eval_results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "benchmark_summary.json")
REPORT_FILE = os.path.join(BASE_DIR, "analysis", "findings_report.md")


@st.cache_resource
def load_judges():
    return LLMJudgeA(), LLMJudgeB(), ClassicalMLJudge()


@st.cache_data
def load_eval_data():
    if not (os.path.exists(RESULTS_CSV) and os.path.exists(SUMMARY_JSON)):
        from analysis.correlation_analysis import run_full_benchmark

        run_full_benchmark()

    if os.path.exists(RESULTS_CSV) and os.path.exists(SUMMARY_JSON):
        df_res = pd.read_csv(RESULTS_CSV)
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return df_res, summary
    return None, None


judge_a, judge_b, ml_judge = load_judges()
df_res, summary_data = load_eval_data()


# App Header
st.markdown(
    "<div class='main-title'>JudgeBench — LLM-as-a-Judge Evaluation Harness</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-title'>Benchmarking LLM Judges (Gemini 3.1 Flash Lite, Groq Llama 3.3) vs. Classical ML Judge & Hand-Labeled Gold Set</div>",
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Benchmark Module:",
    [
        "Live Listing Inspector",
        "Evaluation & Correlation Study",
        "ML Judge Tuning Log",
        "Disagreement & Failure Analysis",
        "Strategic Findings Report",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Judges Benchmarked:**")
st.sidebar.markdown("- **Judge A:** Gemini 3.1 Flash Lite")
st.sidebar.markdown("- **Judge B:** Groq Llama 3.3 70B")
st.sidebar.markdown("- **ML Judge:** Dual-head LogReg + XGBoost")


# -------------------------------------------------------------
# Module 1: Live Listing Inspector
# -------------------------------------------------------------
if page == "Live Listing Inspector":
    st.header("Live Listing Inspector & Side-by-Side Evaluation")
    st.markdown(
        "Test custom product listings or select preset examples to see all three judges evaluate in real-time."
    )

    preset = st.selectbox(
        "Choose a Preset Test Case:",
        [
            "Custom Input",
            "Preset 1: Clean & Compliant Listing",
            "Preset 2: Material Hallucination (Gold/Cashmere)",
            "Preset 3: Policy Violation (Banned Superlatives)",
            "Preset 4: Title Length Bound Violation (>200 Chars)",
        ],
    )

    if preset == "Preset 1: Clean & Compliant Listing":
        default_brand = "Generic"
        default_cat = "Office Products"
        default_bullets = "Heavy duty steel construction | 50 sheet capacity | Ergonomic non-slip base"
        default_title = "Heavy Duty Steel Stapler - 50 Sheet Capacity, Black"
        default_desc = "Built with premium solid steel for high-volume office use. Features non-slip base and easy top loading."
    elif preset == "Preset 2: Material Hallucination (Gold/Cashmere)":
        default_brand = "Solimo"
        default_cat = "Apparel"
        default_bullets = "100% Breathable Cotton | Machine washable | Classic fit"
        default_title = (
            "Men's T-Shirt Crafted from 100% Pure Italian Cashmere & Solid Gold Threads"
        )
        default_desc = "Luxury grade space-age material imported directly from Italy. Fits all body shapes."
    elif preset == "Preset 3: Policy Violation (Banned Superlatives)":
        default_brand = "Rivindex"
        default_cat = "Home Kitchen"
        default_bullets = "Stainless steel blade | Wooden handle"
        default_title = (
            "WORLD'S BEST #1 GUARANTEED CHEAP SALE STAINLESS STEEL CHEF KNIFE!!!"
        )
        default_desc = "Unbeatable miracle knife! Guaranteed 100% to cure kitchen fatigue and make cooking 10x faster!"
    elif preset == "Preset 4: Title Length Bound Violation (>200 Chars)":
        default_brand = "Stone & Beam"
        default_cat = "Furniture"
        default_bullets = "Solid oak legs | Linen fabric cushion"
        default_title = "Modern Armchair Living Room Chair Accent Chair Soft Fabric Linen Cushion Solid Wood Frame Heavy Duty Durable Best Seller Cheap Discount Sale Special Promo Item Model Home Furniture"
        default_desc = (
            "Overly long title designed for keyword stuffing search manipulation."
        )
    else:
        default_brand = "Generic"
        default_cat = "Electronics"
        default_bullets = "1.75mm diameter | 1kg spool"
        default_title = "PETG 3D Printer Filament, 1.75mm, 1 kg Spool"
        default_desc = "High clarity PETG filament for reliable 3D printing."

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        brand = st.text_input("Product Brand:", value=default_brand)
        category = st.text_input("Product Category:", value=default_cat)
        source_bullets = st.text_area(
            "Source Attributes (Pipe separated):", value=default_bullets
        )

    with col_in2:
        generated_title = st.text_input("Generated Title:", value=default_title)
        generated_description = st.text_area(
            "Generated Description:", value=default_desc
        )

    if st.button("Run All 3 Judges", type="primary"):
        input_listing = {
            "brand": brand,
            "category": category,
            "source_bullets": [b.strip() for b in source_bullets.split("|")],
            "generated_title": generated_title,
            "generated_description": generated_description,
        }

        with st.spinner(
            "Evaluating with Gemini 3.1 Flash Lite, Groq Llama 3.3, and ML Judge..."
        ):
            res_a = judge_a.evaluate(input_listing)
            res_b = judge_b.evaluate(input_listing)
            res_ml = ml_judge.evaluate(input_listing)

        st.markdown("### Side-by-Side Verdict Matrix")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='judge-card'>", unsafe_allow_html=True)
            st.markdown("#### LLM Judge A")
            st.markdown("`Gemini 3.1 Flash Lite`")
            comp_badge = (
                "<span class='badge-pass'>PASS</span>"
                if res_a["compliance_pass"]
                else "<span class='badge-fail'>FAIL</span>"
            )
            st.markdown(f"**Compliance Status:** {comp_badge}", unsafe_allow_html=True)
            st.markdown(f"**Quality Score:** `{res_a['quality_score']:.1f} / 5.0`")
            st.markdown(f"**Fact Accuracy:** `{res_a['accuracy_score']:.1f} / 5.0`")
            st.markdown(f"**Reasoning:** *{res_a['reasoning']}*")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='judge-card'>", unsafe_allow_html=True)
            st.markdown("#### LLM Judge B")
            st.markdown("`Groq Llama 3.3 70B`")
            comp_badge = (
                "<span class='badge-pass'>PASS</span>"
                if res_b["compliance_pass"]
                else "<span class='badge-fail'>FAIL</span>"
            )
            st.markdown(f"**Compliance Status:** {comp_badge}", unsafe_allow_html=True)
            st.markdown(f"**Quality Score:** `{res_b['quality_score']:.1f} / 5.0`")
            st.markdown(f"**Fact Accuracy:** `{res_b['accuracy_score']:.1f} / 5.0`")
            st.markdown(f"**Reasoning:** *{res_b['reasoning']}*")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='judge-card'>", unsafe_allow_html=True)
            st.markdown("#### Classical ML Judge")
            st.markdown("`LogReg + XGBoost Regressor`")
            comp_badge = (
                "<span class='badge-pass'>PASS</span>"
                if res_ml["compliance_pass"]
                else "<span class='badge-fail'>FAIL</span>"
            )
            st.markdown(f"**Compliance Status:** {comp_badge}", unsafe_allow_html=True)
            st.markdown(f"**Quality Score:** `{res_ml['quality_score']:.1f} / 5.0`")
            st.markdown(f"**Fact Accuracy:** `{res_ml['accuracy_score']:.1f} / 5.0`")
            st.markdown(f"**Reasoning:** *{res_ml['reasoning']}*")
            st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Module 2: Evaluation & Correlation Study
# -------------------------------------------------------------
elif page == "Evaluation & Correlation Study":
    st.header("Evaluation Benchmark & Correlation Analysis")

    if summary_data and df_res is not None:
        corr = summary_data["correlations"]
        comp = summary_data["compliance_metrics"]

        st.markdown("### Key Performance Indicators (N = 75 Gold Standard Examples)")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(
            "Inter-LLM Agreement (Spearman)",
            f"{corr['spearman_judge_a_vs_b']:.4f}",
            "Gemini vs Llama",
        )
        col_m2.metric(
            "Judge A vs. Gold Spearman",
            f"{corr['spearman_judge_a_vs_gold']:.4f}",
            "Gemini 3.1 Flash",
        )
        col_m3.metric(
            "Judge B vs. Gold Spearman",
            f"{corr['spearman_judge_b_vs_gold']:.4f}",
            "Groq Llama 3.3",
        )
        col_m4.metric(
            "ML Classifier Compliance F1",
            f"{comp['Classical_ML']['f1_score']:.4f}",
            "Tuned LogisticReg",
        )

        st.markdown("---")
        st.markdown("### Inter-Judge Quality Score Agreement Scatter Plot")

        fig_scatter = px.scatter(
            df_res,
            x="judge_a_quality",
            y="judge_b_quality",
            color="intended_flaw",
            hover_data=["generated_title", "human_quality_score"],
            labels={
                "judge_a_quality": "LLM Judge A Score (Gemini)",
                "judge_b_quality": "LLM Judge B Score (Llama 3.3)",
            },
            title="LLM Judge A vs. LLM Judge B Quality Score Correlation",
            template="plotly_white",
        )
        fig_scatter.add_shape(
            type="line", x0=1, y0=1, x1=5, y1=5, line=dict(color="gray", dash="dash")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Catalog Policy Compliance Classification Performance")

        df_comp = pd.DataFrame(
            [
                {"Judge": "LLM Judge A (Gemini 3.1 Flash Lite)", **comp["LLM_Judge_A"]},
                {"Judge": "LLM Judge B (Groq Llama 3.3)", **comp["LLM_Judge_B"]},
                {"Judge": "Classical ML Classifier (Tuned)", **comp["Classical_ML"]},
            ]
        )
        st.table(
            df_comp.style.format(
                {
                    "accuracy": "{:.2%}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1_score": "{:.4f}",
                }
            )
        )

    else:
        st.warning(
            "Benchmark results not found on disk. Run `python analysis/correlation_analysis.py` to populate data."
        )


# -------------------------------------------------------------
# Module 3: ML Judge Tuning Log
# -------------------------------------------------------------
elif page == "ML Judge Tuning Log":
    st.header("Classical ML Judge Hyperparameter Tuning Log")
    st.markdown(
        "Demonstrating systematic hyperparameter optimization via `GridSearchCV` on the dual-head Classical ML Judge."
    )

    if summary_data and "tuning_log" in summary_data:
        tune = summary_data["tuning_log"]

        st.markdown("### 1. Classification Head (Policy Compliance Gate)")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.json(tune["classification_head"])
        with col_c2:
            fig_cls = go.Figure(
                data=[
                    go.Bar(
                        name="Before Tuning",
                        x=["F1-Score", "Accuracy"],
                        y=[
                            tune["classification_head"]["before_tuning"]["f1_score"],
                            tune["classification_head"]["before_tuning"]["accuracy"],
                        ],
                    ),
                    go.Bar(
                        name="After GridSearch",
                        x=["F1-Score", "Accuracy"],
                        y=[
                            tune["classification_head"]["after_tuning"]["f1_score"],
                            tune["classification_head"]["after_tuning"]["accuracy"],
                        ],
                    ),
                ]
            )
            fig_cls.update_layout(
                title="Classifier Metric Improvement",
                barmode="group",
                template="plotly_white",
            )
            st.plotly_chart(fig_cls, use_container_width=True)

        st.markdown("### 2. Regression Head (Continuous Quality Predictor)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.json(tune["regression_head"])
        with col_r2:
            fig_reg = go.Figure(
                data=[
                    go.Bar(
                        name="Before Tuning",
                        x=["MSE", "Spearman Correlation"],
                        y=[
                            tune["regression_head"]["before_tuning"]["mse"],
                            tune["regression_head"]["before_tuning"]["spearman"],
                        ],
                    ),
                    go.Bar(
                        name="After GridSearch",
                        x=["MSE", "Spearman Correlation"],
                        y=[
                            tune["regression_head"]["after_tuning"]["mse"],
                            tune["regression_head"]["after_tuning"]["spearman"],
                        ],
                    ),
                ]
            )
            fig_reg.update_layout(
                title="Regressor Metric Improvement",
                barmode="group",
                template="plotly_white",
            )
            st.plotly_chart(fig_reg, use_container_width=True)


# -------------------------------------------------------------
# Module 4: Disagreement & Failure Analysis
# -------------------------------------------------------------
elif page == "Disagreement & Failure Analysis":
    st.header("Disagreement & Qualitative Failure Analysis")
    st.markdown(
        "Examining key edge cases where LLM Judges disagreed with Human Ground Truth or with each other."
    )

    if summary_data and "disagreement_cases" in summary_data:
        cases = summary_data["disagreement_cases"]
        for idx, case in enumerate(cases):
            with st.expander(
                f"Case #{idx + 1}: {case['generated_title'][:70]}... (Flaw: {case['intended_flaw']})"
            ):
                st.markdown(f"**Full Generated Title:** {case['generated_title']}")
                st.markdown(
                    f"**Ground Truth Quality:** `{case['human_quality_score']}` | **Judge A Quality:** `{case['judge_a_quality']}` | **Judge B Quality:** `{case['judge_b_quality']}`"
                )
                st.markdown(f"**Judge A Reasoning:** *{case['judge_a_reasoning']}*")
                st.markdown(f"**Judge B Reasoning:** *{case['judge_b_reasoning']}*")
    else:
        st.info(
            "Run `python analysis/correlation_analysis.py` to extract disagreement cases."
        )


# -------------------------------------------------------------
# Module 5: Findings Report
# -------------------------------------------------------------
elif page == "Strategic Findings Report":
    st.header("Strategic Findings & Architecture Recommendations")
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.error("Report file not found at analysis/findings_report.md.")
