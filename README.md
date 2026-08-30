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

## Quick Start

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
- **Fold checkpointing**: Training saves every fold model — resume on crash
- **Demo mode**: Dashboard works without trained model (heuristic scoring)
- **MLOps narrative**: Model monitoring, drift detection, A/B testing strategy

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | XGBoost, LightGBM |
| Explainability | SHAP |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |

## MLOps Considerations

- **CI/CD**: Model retraining via GitHub Actions on new fraud labels
- **Model Monitoring**: Track prediction drift with Evidently AI
- **Feature Store**: Pre-computed aggregations in Feast/Tecton
- **A/B Testing**: Shadow-score new model versions before promotion
- **Data Versioning**: DVC for dataset lineage tracking

---

**RazorPay AI Buildathon 2026** | Post-Onboarding Merchant Risk Detection
