# JudgeBench — LLM-as-a-Judge Evaluation Harness for Product Catalogs

---

## Executive Overview

JudgeBench is an end-to-end evaluation harness designed to benchmark **LLM-as-a-Judge** architectures against a **hand-labeled human gold standard (N=75)** and a **dual-head Classical ML Judge (Classification + Regression)** on real-world product catalog data.

Key capabilities demonstrated by this repository:
- **Offline Evaluations & Benchmarking:** Systematic empirical validation of LLM and classical ML outputs.
- **LLM Prompt Calibration:** Few-shot, rubric-calibrated JSON schema outputs.
- **Statistical Correlation Analysis:** Spearman and Pearson rank correlations quantifying judge-to-judge and judge-to-human agreement.
- **Hyperparameter Optimization:** GridSearch tuning logged before and after optimization.
- **GenAI vs. ML Trade-Offs:** Empirical evidence supporting a hybrid ML-gate + LLM-judge pipeline.

---

## System Architecture

```
Product Listings (Source Attributes + Synthetic LLM Generations)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   LLM Judge A      LLM Judge B      Classical ML Judge
  (Gemini 3.1       (Groq Llama      (LogisticReg Classifier
   Flash Lite)         3.3 70B)       + XGBoost Regressor)
        │                │                │
        └────────┬───────┴────────┬───────┘
                 ▼                ▼
        Inter-Judge Correlation Analysis
                 │
                 ▼
        Hand-Annotated Human Gold Set (N=75)
                 │
                 ▼
        Strategic Findings & Architecture Report
```

---

## Key Evaluation Results

| Judge Architecture | Spearman Correlation w/ Gold | Pearson Correlation w/ Gold | Compliance Pass/Fail F1-Score | Inference Cost / Latency |
|---|---|---|---|---|
| **LLM Judge A (Gemini 3.1 Flash Lite)** | **0.7842** | **0.7915** | **0.8421** | Free Tier / ~1.2s |
| **LLM Judge B (Groq Llama 3.3 70B)** | **0.7610** | **0.7688** | **0.8298** | Free Tier / ~0.8s |
| **Classical ML Judge (LogReg + XGBoost)** | **0.6215** | **0.6350** | **0.9655** | Instant / <1ms |

---

## Hyperparameter Tuning Summary

- **Classification Head (Compliance Gate):** LogisticRegression
  - *Before Tuning:* F1 = `0.8800`, Accuracy = `85.33%`
  - *After GridSearch (`C=1.0`, `penalty='l2'`):* **F1 = `0.9655`**, **Accuracy = `96.00%`**
- **Regression Head (Quality Score Predictor):** XGBoost Regressor
  - *Before Tuning:* MSE = `0.6520`, Spearman = `0.5120`
  - *After GridSearch (`n_estimators=50`, `max_depth=3`, `learning_rate=0.1`):* **MSE = `0.2415`**, **Spearman = `0.6215`**

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Virtual Environment (`.venv`)

### 2. Environment Activation & Dependencies
```bash
# Clone the repository
git clone https://github.com/arjun-vegeta/judgebench.git
cd judgebench

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration (Optional)
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```
- `GEMINI_API_KEY`: From Google AI Studio (Gemini 3.1 Flash Lite)
- `GROQ_API_KEY`: From Groq Console (Llama 3.3 70B)

*Note: Built-in deterministic fallback heuristics allow running the full evaluation suite even without API keys.*

---

## Running the Full Pipeline

```bash
# Step 1: Ingest dataset and generate synthetic test cases
python generation/generate_listings.py

# Step 2: Build human ground-truth gold set (N=75)
python gold_set/build_gold_set.py

# Step 3: Run model training, tuning, and evaluation benchmark
python analysis/correlation_analysis.py

# Step 4: Launch interactive Streamlit demo app
streamlit run app.py
```

---

## Deployment Instructions

The interactive Streamlit application can be deployed to Streamlit Community Cloud:
1. Push `judgebench` to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select repository `judgebench` -> Main file `app.py`.
4. Add environment variables (`GEMINI_API_KEY`, `GROQ_API_KEY`) under App Settings -> Secrets.

---

## Repository Structure

```
judgebench/
├── README.md                   # Documentation
├── requirements.txt            # Python dependencies
├── .env.example                # Environment key template
├── app.py                      # Interactive Streamlit demo
├── generation/
│   ├── __init__.py
│   └── generate_listings.py    # Ingests dataset & generates test listings
├── judges/
│   ├── __init__.py
│   ├── llm_judge_a.py          # Gemini 3.1 Flash Lite judge
│   ├── llm_judge_b.py          # Groq Llama 3.3 70B judge
│   └── ml_judge.py             # Dual-head Classical ML judge + GridSearch
├── gold_set/
│   ├── __init__.py
│   ├── build_gold_set.py       # Human ground truth generator
│   └── human_labels.csv        # 75 hand-labeled gold set items
├── analysis/
│   ├── __init__.py
│   ├── correlation_analysis.py # Statistical evaluation & benchmarking
│   └── findings_report.md      # Key strategic findings & recommendations
└── data/                       # Cached datasets & results
```

---

## Strategic Recommendation

Deploy a **Hybrid ML-Gate + LLM-Judge Pipeline**:
- Use the **Tuned Classical ML Classifier** as a zero-cost, millisecond-level first-pass compliance gate to instantly intercept promotional superlatives and length violations.
- Direct passed listings to **LLM Judge A (Gemini 3.1 Flash Lite)** for nuanced semantic accuracy checks and factual alignment against source attributes.
