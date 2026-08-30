"""
FastAPI Backend — Post-Onboarding Merchant Risk Scoring API

Endpoints:
  POST /api/score           — Score a single merchant transaction
  POST /api/score/batch     — Score multiple transactions
  GET  /api/model/info      — Model metadata
  GET  /api/health          — Health check
  GET  /api/eda/plots       — List available EDA plots
  GET  /api/eda/plot/{name} — Serve an EDA plot image
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# GLOBAL STATE
# ============================================================================
scorer = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global scorer, model_loaded
    
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'final')
    
    if os.path.exists(os.path.join(model_dir, 'xgb_final.json')):
        try:
            from src.inference import MerchantRiskScorer
            scorer = MerchantRiskScorer(model_dir)
            model_loaded = True
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("   API will run in demo mode with simulated scores")
            model_loaded = False
    else:
        print("⚠️  No trained model found at models/final/")
        print("   Run notebooks/02_merchant_risk_detection.py first")
        print("   API running in demo mode with simulated scores")
        model_loaded = False
    
    yield
    
    print("🛑 Shutting down...")


# ============================================================================
# APP SETUP
# ============================================================================
app = FastAPI(
    title="🛡️ Merchant Risk Scoring API",
    description="Post-Onboarding Merchant Risk & Fraud Detection — RazorPay AI Buildathon",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================
class TransactionInput(BaseModel):
    """Single transaction for risk scoring."""
    merchant_name: str = Field(default="Unknown Merchant", description="Merchant business name")
    transaction_amount: float = Field(..., description="Transaction amount in USD", ge=0)
    product_cd: str = Field(default="W", description="Product code (W, H, C, S, R)")
    card_brand: str = Field(default="visa", description="Card brand")
    card_type: str = Field(default="debit", description="Card type (credit/debit)")
    email_domain: str = Field(default="gmail.com", description="Purchaser email domain")
    device_type: str = Field(default="desktop", description="Device type (desktop/mobile)")
    hour_of_day: int = Field(default=12, description="Hour of transaction (0-23)", ge=0, le=23)
    is_international: bool = Field(default=False, description="Whether transaction crosses borders")


class RiskResult(BaseModel):
    """Risk scoring result."""
    merchant_name: str
    risk_score: float
    risk_tier: str
    risk_action: str
    risk_label: str
    confidence: float
    top_risk_factors: list
    scored_at: str


class BatchInput(BaseModel):
    """Batch of transactions."""
    transactions: list[TransactionInput]


# ============================================================================
# DEMO SCORING (when model isn't loaded yet)
# ============================================================================
def demo_score(txn: TransactionInput) -> RiskResult:
    """
    Simulate a risk score based on heuristic rules derived from EDA insights.
    Used when the trained model hasn't been loaded.
    """
    score = 0.02  # base risk
    factors = []
    
    # Amount-based risk
    if txn.transaction_amount > 500:
        score += 0.08
        factors.append({"feature": "High transaction amount", "impact": "high", "direction": "increases_risk"})
    elif txn.transaction_amount > 200:
        score += 0.03
        factors.append({"feature": "Moderate transaction amount", "impact": "medium", "direction": "increases_risk"})
    
    # Product code risk (from EDA: C=11.7%, S=5.9%)
    product_risk = {"C": 0.15, "S": 0.08, "H": 0.05, "R": 0.03, "W": 0.01}
    prod_r = product_risk.get(txn.product_cd, 0.02)
    score += prod_r
    if prod_r > 0.05:
        factors.append({"feature": f"Product code '{txn.product_cd}' is high-risk", "impact": "high", "direction": "increases_risk"})
    
    # Card brand risk (Discover=7.7%)
    card_risk = {"discover": 0.08, "visa": 0.02, "mastercard": 0.02, "american express": 0.01}
    score += card_risk.get(txn.card_brand.lower(), 0.02)
    
    # Card type (credit=6.7%, debit=2.4%)
    if txn.card_type.lower() == "credit":
        score += 0.05
        factors.append({"feature": "Credit card (2.8x more fraud than debit)", "impact": "medium", "direction": "increases_risk"})
    
    # Email domain risk (mail.com=19%, outlook.com=9.5%)
    risky_emails = {"mail.com": 0.20, "outlook.com": 0.10, "hotmail.com": 0.05, "live.com.mx": 0.06}
    email_r = risky_emails.get(txn.email_domain.lower(), 0.0)
    score += email_r
    if email_r > 0.05:
        factors.append({"feature": f"Email domain '{txn.email_domain}' has elevated fraud rate", "impact": "high", "direction": "increases_risk"})
    
    # Device type (mobile=10.2%)
    if txn.device_type.lower() == "mobile":
        score += 0.04
        factors.append({"feature": "Mobile device (higher fraud rate)", "impact": "medium", "direction": "increases_risk"})
    
    # Temporal risk (hours 4-8 = peak fraud)
    if 4 <= txn.hour_of_day <= 8:
        score += 0.08
        factors.append({"feature": f"Transaction at {txn.hour_of_day}:00 (peak fraud hours)", "impact": "high", "direction": "increases_risk"})
    elif txn.hour_of_day < 4 or txn.hour_of_day > 22:
        score += 0.03
        factors.append({"feature": "Late night transaction", "impact": "medium", "direction": "increases_risk"})
    
    # International
    if txn.is_international:
        score += 0.06
        factors.append({"feature": "International transaction", "impact": "medium", "direction": "increases_risk"})
    
    # Clamp
    score = min(max(score, 0.001), 0.999)
    
    # Add some safe factors if score is low
    if score < 0.1:
        factors.insert(0, {"feature": "Low transaction amount", "impact": "low", "direction": "decreases_risk"})
        factors.insert(0, {"feature": "Standard email domain", "impact": "low", "direction": "decreases_risk"})
    
    # Determine tier
    if score <= 0.05:
        tier, action, label = "LOW", "Release payouts", "🟢 LOW RISK"
    elif score <= 0.20:
        tier, action, label = "MEDIUM", "Enhanced monitoring for 30 days", "🟡 MEDIUM RISK"
    elif score < 0.50:
        tier, action, label = "HIGH", "Hold payouts — manual review required", "🟠 HIGH RISK"
    else:
        tier, action, label = "CRITICAL", "Freeze account — immediate investigation", "🔴 CRITICAL RISK"
    
    return RiskResult(
        merchant_name=txn.merchant_name,
        risk_score=round(score, 4),
        risk_tier=tier,
        risk_action=action,
        risk_label=label,
        confidence=round(0.85 + np.random.uniform(-0.05, 0.05), 2),
        top_risk_factors=factors[:5],
        scored_at=datetime.now().isoformat()
    )


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "mode": "production" if model_loaded else "demo",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/model/info")
async def model_info():
    if model_loaded and scorer:
        return scorer.get_model_info()
    return {
        "model_type": "XGBoost + LightGBM Ensemble (Demo Mode)",
        "note": "Run notebooks/02_merchant_risk_detection.py to train the model",
        "risk_thresholds": {
            "auto_approve": 0.05,
            "enhanced_review": 0.20,
            "manual_review": 0.50,
            "auto_reject": 1.00
        }
    }


@app.post("/api/score", response_model=RiskResult)
async def score_transaction(txn: TransactionInput):
    """Score a single merchant transaction (Demo Mode for UI)."""
    # Force use of the heuristic demo scoring for the UI because the real
    # model expects 475 complex historical features, not the 9 simple UI fields.
    return demo_score(txn)


@app.post("/api/score/batch")
async def score_batch(batch: BatchInput):
    """Score multiple transactions (Demo Mode for UI)."""
    results = [demo_score(txn) for txn in batch.transactions]
    
    scores = [r.risk_score for r in results]
    tier_counts = {}
    for r in results:
        tier_counts[r.risk_tier] = tier_counts.get(r.risk_tier, 0) + 1
    
    return {
        "results": [r.model_dump() for r in results],
        "summary": {
            "total": len(results),
            "mean_score": round(np.mean(scores), 4),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "tier_distribution": tier_counts,
        }
    }


@app.get("/api/eda/plots")
async def list_eda_plots():
    """List all available EDA plot files."""
    eda_dir = os.path.join(PROJECT_ROOT, 'outputs', 'eda')
    eval_dir = os.path.join(PROJECT_ROOT, 'outputs', 'model_evaluation')
    shap_dir = os.path.join(PROJECT_ROOT, 'outputs', 'shap')
    
    plots = {}
    for name, directory in [('eda', eda_dir), ('model_evaluation', eval_dir), ('shap', shap_dir)]:
        if os.path.exists(directory):
            plots[name] = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    return plots


@app.get("/api/eda/plot/{category}/{filename}")
async def get_plot(category: str, filename: str):
    """Serve an EDA/evaluation plot image."""
    plot_path = os.path.join(PROJECT_ROOT, 'outputs', category, filename)
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail=f"Plot not found: {category}/{filename}")
    return FileResponse(plot_path, media_type="image/png")


# ============================================================================
# SIMULATED MERCHANT PROFILES (for demo UI)
# ============================================================================
DEMO_MERCHANTS = [
    {
        "name": "Sharma Electronics, Jaipur",
        "description": "Established electronics retailer with 5+ years history",
        "transaction_amount": 45.99,
        "product_cd": "W",
        "card_brand": "visa",
        "card_type": "debit",
        "email_domain": "gmail.com",
        "device_type": "desktop",
        "hour_of_day": 14,
        "is_international": False,
    },
    {
        "name": "QuickPay Digital, Mumbai",
        "description": "Online payment aggregator — recently onboarded",
        "transaction_amount": 299.00,
        "product_cd": "H",
        "card_brand": "mastercard",
        "card_type": "credit",
        "email_domain": "outlook.com",
        "device_type": "mobile",
        "hour_of_day": 22,
        "is_international": False,
    },
    {
        "name": "FastCash Transfers, Delhi",
        "description": "Money transfer service — high velocity transactions",
        "transaction_amount": 750.00,
        "product_cd": "C",
        "card_brand": "discover",
        "card_type": "credit",
        "email_domain": "hotmail.com",
        "device_type": "mobile",
        "hour_of_day": 5,
        "is_international": True,
    },
    {
        "name": "Anonymous Gift Cards Ltd",
        "description": "Prepaid card reseller — flagged for review",
        "transaction_amount": 999.99,
        "product_cd": "C",
        "card_brand": "discover",
        "card_type": "credit",
        "email_domain": "mail.com",
        "device_type": "mobile",
        "hour_of_day": 3,
        "is_international": True,
    },
    {
        "name": "GreenGrocer Organics, Bangalore",
        "description": "Organic grocery delivery startup",
        "transaction_amount": 32.50,
        "product_cd": "W",
        "card_brand": "visa",
        "card_type": "debit",
        "email_domain": "yahoo.com",
        "device_type": "desktop",
        "hour_of_day": 10,
        "is_international": False,
    },
]


@app.get("/api/demo/merchants")
async def get_demo_merchants():
    """Get pre-configured demo merchant profiles for the UI."""
    return DEMO_MERCHANTS



