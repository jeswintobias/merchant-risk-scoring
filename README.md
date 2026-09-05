# 🛡️ Post-Onboarding Merchant Risk & Fraud Detection

> **RazorPay AI Buildathon 2026** — AI Risk Manager Track

## Problem Statement

Onboarding filters (KYC, business registration) only verify if a business legally exists—they cannot predict whether a merchant will engage in fraudulent processing once active. Bad actors often pass initial KYC checks, onboard onto payment gateways, and quickly process fraudulent, stolen-card, or chargeback-heavy transactions before absconding with payouts.

**Our Solution**: An Early-Lifecycle Transaction Monitoring Engine that continuously analyses merchant transaction streams post-onboarding, aggregates risk signals, and automatically triggers mitigations (payout holds, reviews, account freezes).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│           EARLY-LIFECYCLE MERCHANT RISK ENGINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. TRANSACTION STREAM     2. BEHAVIORAL PROFILING   3. RISK SCORING │
│     • Real-time txns          • Velocity features       • XGBoost    │
│     • Card/device data        • Amount volatility       • LightGBM   │
│     • Identity signals        • Card diversity          • Ensemble   │
│                               • Device entropy                       │
│                               • Temporal patterns     4. ACTION      │
│                                                         • Release ✅ │
│                                                         • Monitor 🟡 │
│                                                         • Hold 🟠    │
│                                                         • Freeze 🔴  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🏆 Track 02 — AI Risk Manager Highlights
This project was built explicitly to satisfy the **RazorPay Track 02** requirements:
1. **The LLM Auto-Responder**: We integrated Google's **Gemini 1.5 Pro** LLM to act as an AI Risk Manager. When a high/critical risk transaction is detected, the system passes SHAP risk factors into Gemini to automatically draft a professional payout-hold email requesting specific verification documents (e.g. shipping invoices).
2. **False-Positive Cost Metrics (ROI)**: We built a dashboard that calculates our model's precision against the manual review cost of false positives, proving the financial viability of our detector on a held-out test set.
3. **Engineering Maturity**: The entire solution is Dockerized, equipped with a Fast API backend and a Vite+React frontend, and utilizes an XGBoost + LightGBM ensemble.

*Note for Judges: If you prefer not to use Docker, follow the Local Setup instructions below.*

## Quick Start (Docker)

**Prerequisite: API Key**
Since the project relies on Gemini 1.5 Pro for the AI Auto-Responder, you must provide your own API key. 
1. Create a `.env` file in the root directory.
2. Add the following line: `GEMINI_API_KEY=your_api_key_here`

The easiest way to run the entire stack (Backend + Frontend) is via Docker Compose:

```bash
docker-compose up --build
```
- Dashboard: http://localhost:5173
- API / Docs: http://localhost:8000/docs

## Quick Start (Local Setup)

If you don't want to use Docker, you can run the app natively in under a minute using the `requirements.txt` fallback.

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run EDA (Optional — validate plots)
```bash
MPLBACKEND=Agg python notebooks/01_eda.py
# Plots saved to outputs/eda/
```

### 3. Train Models
```bash
MPLBACKEND=Agg python notebooks/02_merchant_risk_detection.py
# Models saved to models/final/ with checkpoints in models/checkpoints/
```

### 4. Start Dashboard
```bash
# Terminal 1 — Backend
source .venv/bin/activate
uvicorn app.backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd app/frontend
npm install
npm run dev
```

Open http://localhost:5173 for the dashboard.

## Project Structure

```
├── data/                          # IEEE-CIS Fraud Detection dataset
│   ├── demo_batch.csv             # Demo data for batch testing
│   └── ieee-fraud-detection/
├── notebooks/
│   ├── 01_eda.py                  # Standalone EDA (run first)
│   └── 02_merchant_risk_detection.py  # Full training pipeline
├── src/
│   ├── data_loader.py             # Memory-optimised data loading
│   ├── feature_engineering.py     # 47+ feature engineering pipeline
│   ├── model_utils.py             # Checkpointing, model saving
│   └── inference.py               # MerchantRiskScorer class
├── app/
│   ├── backend/main.py            # FastAPI REST API
│   └── frontend/                  # React + Vite dashboard
├── models/
│   ├── final/                     # Production models + model card
│   ├── checkpoints/               # Per-fold model checkpoints
│   └── artifacts/                 # Feature encoders, lists
├── outputs/
│   ├── eda/                       # 9 validated EDA plots
│   ├── model_evaluation/          # Performance dashboard, confusion matrices
│   └── shap/                      # SHAP explainability plots
├── submissions/                   # Kaggle submission CSV
└── requirements.txt
```

## Dataset

**IEEE-CIS Fraud Detection** (Vesta Corporation)
- 590,540 training transactions | 506,691 test transactions
- 394 transaction + 41 identity features
- 3.50% fraud rate (28:1 class imbalance)

Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
Place in `data/ieee-fraud-detection/`

## Models

| Model | Mean AUC | Training |
|-------|----------|----------|
| XGBoost | ~0.95 | GroupKFold × 6 by month |
| LightGBM | ~0.95 | GroupKFold × 6 by month |
| **Ensemble** | **~0.96** | Weighted blend |

### Risk Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0.00 – 0.05 | 🟢 LOW | Release payouts |
| 0.05 – 0.20 | 🟡 MEDIUM | Enhanced monitoring (30 days) |
| 0.20 – 0.50 | 🟠 HIGH | Hold payouts — manual review |
| 0.50 – 1.00 | 🔴 CRITICAL | Freeze account immediately |

## Key Features

- **47+ engineered features**: UID-based velocity, amount stats, device entropy
- **SHAP explainability**: Every risk decision is explainable (RBI-compliant)
- **Batch CSV Analysis**: Process bulk historical records for instant portfolio risk assessment
- **Fold checkpointing**: Training saves every fold model — resume on crash
- **Demo mode**: Dashboard works without trained model (heuristic scoring)
- **MLOps narrative**: Model monitoring, drift detection, A/B testing strategy

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | XGBoost, LightGBM |
| Explainability | SHAP |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite + Tailwind CSS v4 + shadcn/ui |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |

## MLOps Considerations

- **CI/CD**: Model retraining via GitHub Actions on new fraud labels
- **Model Monitoring**: Track prediction drift with Evidently AI
- **Feature Store**: Pre-computed aggregations in Feast/Tecton
- **A/B Testing**: Shadow-score new model versions before promotion
- **Data Versioning**: DVC for dataset lineage tracking

## Build Challenges & Technical Obstacles

1. **Dataset Discovery & Class Imbalance**: Finding a high-quality, realistic financial dataset was initially challenging. Once secured, we faced extreme class imbalance (fraud is rare). I had to rely on advanced Tree-Based Ensembling (XGBoost + LightGBM) to capture rare fraud signals without overfitting.
2. **Transitioning from Jupyter to MLOps**: I initially built the prototype in a single `.ipynb` notebook. When the model crashed during a long training run, I lost all my progress because I hadn't implemented model checkpoints. This was a hard lesson in MLOps, forcing me to modularize the codebase into proper folder structures (`src/`, `models/`, `app/`) and implement persistent model saving.
3. **LLM Context Limits (The SHAP Solution)**: Feeding a raw transaction with 434 columns to an LLM to generate an email is highly inefficient and causes hallucinations. I overcame this by integrating SHAP (Explainable AI) to mathematically extract only the top 5 risk-driving features, drastically reducing the LLM prompt size while increasing email accuracy.
4. **Docker Architecture & System Incompatibilities**: Containerizing a heavy ML stack on Apple Silicon presented significant challenges. I encountered a severe serialization bug where XGBoost models trained natively on Python 3.12 crashed when loaded inside a Python 3.10 Docker container. This required debugging multi-architecture base images and strictly aligning environment versions across the host and container to achieve a flawless production build.

## References

1. Killeen, B., Tran, M. T., & Pakana, F. (2026). Chargeback Fraud Detection on Anonymised Merchant Data: An Industry Case Study.
2. Guo, X., Dong, L., Li, Y., Wang, Y., Zhang, P., & Zhu, Z. (2026). Trust-Aware Enterprise Credit Risk Prediction via Variational Autoencoder in Supply Chain Finance.
3. Raju, C. G., Jagadeesha, N., Sharma, S., Sharma, S., Raj, Y., Shetty, T. G., & Premnath, B. (2026). Merchant Fraud Detection Using Machine Learning on Structured Transaction Data.
4. Cai, X., Dai, W., & Lu, J. (2025). Loan Default Prediction Based on Machine Learning Approaches.
5. He, Y. (2025). Auto Loan Defaults: Predictive Modeling and Key Drivers.

---

**RazorPay AI Buildathon 2026** | Post-Onboarding Merchant Risk Detection
