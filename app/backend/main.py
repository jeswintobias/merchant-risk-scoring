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
import io
import json
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, UploadFile, File
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
feature_list = None
label_encoders = None

# Initialize Gemini
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured.")
else:
    print("⚠️ No GEMINI_API_KEY found. Will use fallback templates.")

# ============================================================================
# FEATURE ADAPTER — Maps 9 UI fields → 434 model features
# ============================================================================
def load_model_artifacts():
    """Load feature list and label encoders for the feature adapter."""
    global feature_list, label_encoders

    artifact_dir = os.path.join(PROJECT_ROOT, 'models', 'artifacts')

    # Load feature list
    feat_path = os.path.join(artifact_dir, 'model_features.json')
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            feature_list = json.load(f)
        print(f"   📋 Feature list loaded: {len(feature_list)} features")

    # Load label encoders for mapping categorical values
    le_path = os.path.join(artifact_dir, 'label_encoders.pkl')
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            label_encoders = pickle.load(f)
        print(f"   🏷️  Label encoders loaded: {len(label_encoders)} columns")


def safe_label_encode(encoder, value):
    """Encode a categorical value, returning -1 for unseen categories."""
    str_val = str(value)
    if str_val in encoder.classes_:
        return int(encoder.transform([str_val])[0])
    return -1


def build_model_features(txn) -> pd.DataFrame:
    """
    Bridge the 9 UI input fields → a DataFrame with all 434 model features.

    Strategy:
    - Map known UI fields to their corresponding model features
    - Label-encode categorical fields using saved encoders
    - Fill all remaining features with -1 (the model's missing-value sentinel)
    - Compute derived features where possible (cents, log-amt, etc.)

    This enables the real model to score UI transactions, though accuracy
    will be lower than scoring with full historical features.
    """
    if feature_list is None:
        return None

    # Start with all features set to -1 (missing sentinel)
    row = {feat: -1.0 for feat in feature_list}

    # --- Direct numeric mappings ---
    row['TransactionAmt'] = float(txn.transaction_amount)

    # Derived amount features
    row['cents'] = float(txn.transaction_amount - np.floor(txn.transaction_amount))
    row['TransactionAmt_log'] = float(np.log1p(txn.transaction_amount))

    # --- Categorical mappings via label encoders ---
    if label_encoders:
        # Product code
        if 'ProductCD' in label_encoders and 'ProductCD' in row:
            row['ProductCD'] = safe_label_encode(label_encoders['ProductCD'], txn.product_cd)

        # Card brand → card4
        if 'card4' in label_encoders and 'card4' in row:
            row['card4'] = safe_label_encode(label_encoders['card4'], txn.card_brand)

        # Card type → card6
        if 'card6' in label_encoders and 'card6' in row:
            row['card6'] = safe_label_encode(label_encoders['card6'], txn.card_type)

        # Email domain
        if 'P_emaildomain' in label_encoders and 'P_emaildomain' in row:
            row['P_emaildomain'] = safe_label_encode(label_encoders['P_emaildomain'], txn.email_domain)

        # Device type
        if 'DeviceType' in label_encoders and 'DeviceType' in row:
            row['DeviceType'] = safe_label_encode(label_encoders['DeviceType'], txn.device_type)

    # --- Temporal signal ---
    # Map hour_of_day to a proxy TransactionDT (model uses DT_hour derived from this)
    # The model doesn't use DT_hour directly (it's removed in feature selection),
    # but temporal aggregation features may benefit from a reasonable value

    # --- Outsider flag ---
    # Without D1 and D15, we can't compute outsider15 reliably, keep as -1

    # Build DataFrame with correct column order
    df = pd.DataFrame([row], columns=feature_list)
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts on startup."""
    global scorer, model_loaded

    model_dir = os.path.join(PROJECT_ROOT, 'models', 'final')

    # Load model artifacts (feature list, encoders) regardless of model
    load_model_artifacts()

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
    scoring_mode: str = "demo"  # "production" or "demo"


class BatchInput(BaseModel):
    """Batch of transactions."""
    transactions: list[TransactionInput]


class AlertRequest(BaseModel):
    merchant_name: str
    transaction_id: str
    amount: float
    risk_tier: str
    risk_factors: list[dict]


# ============================================================================
# APP SETUP
# ============================================================================
app = FastAPI(
    title="🛡️ Merchant Risk Scoring API",
    description="Post-Onboarding Merchant Risk & Fraud Detection — RazorPay AI Buildathon",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: Restrict to localhost origins for security.
# In production, set ALLOWED_ORIGINS env var to your domain(s), comma-separated.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").strip()
if ALLOWED_ORIGINS:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")]
else:
    # Development defaults — only allow localhost
    origins = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alt dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8000",   # Same-origin
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    scoring_mode: str = "demo"  # "production" or "demo"


class BatchInput(BaseModel):
    """Batch of transactions."""
    transactions: list[TransactionInput]


# ============================================================================
# MODEL-BASED SCORING (when real model is loaded)
# ============================================================================
def model_score(txn: TransactionInput) -> RiskResult:
    """
    Score a transaction using the real trained model.

    Maps the 9 UI fields → 434 model features via the feature adapter,
    then runs the XGBoost + LightGBM ensemble.
    """
    # Build the full feature DataFrame
    X = build_model_features(txn)
    if X is None:
        # Fallback if feature list couldn't be loaded
        return demo_score(txn)

    # Score with the real model (with SHAP explanations)
    try:
        result = scorer.score(X, explain=True)
    except Exception as e:
        print(f"⚠️  Model scoring failed: {e}, falling back to demo mode")
        return demo_score(txn)

    risk_score = float(result['risk_scores'][0])
    risk_tier = result['risk_tiers'][0]
    risk_action = result['risk_actions'][0]

    # Build risk label
    tier_labels = {
        'LOW': '🟢 LOW RISK',
        'MEDIUM': '🟡 MEDIUM RISK',
        'HIGH': '🟠 HIGH RISK',
        'CRITICAL': '🔴 CRITICAL RISK',
    }
    risk_label = tier_labels.get(risk_tier, risk_tier)

    # Extract SHAP-based risk factors if available
    top_risk_factors = []
    if 'explanations' in result and result['explanations']:
        explanation = result['explanations'][0]
        for factor in explanation.get('top_factors', [])[:5]:
            top_risk_factors.append({
                "feature": _humanize_feature(factor['feature']),
                "impact": "high" if abs(factor['shap_value']) > 0.1 else "medium",
                "direction": factor['direction'],
            })

    # Add context factors from the UI inputs
    if not top_risk_factors:
        top_risk_factors = _build_heuristic_factors(txn, risk_score)

    # Confidence based on model agreement
    xgb_score = float(result['xgb_scores'][0])
    lgb_score = float(result['lgb_scores'][0])
    agreement = 1.0 - abs(xgb_score - lgb_score)
    confidence = round(min(0.99, max(0.60, agreement)), 2)

    return RiskResult(
        merchant_name=txn.merchant_name,
        risk_score=round(risk_score, 4),
        risk_tier=risk_tier,
        risk_action=risk_action,
        risk_label=risk_label,
        confidence=confidence,
        top_risk_factors=top_risk_factors,
        scored_at=datetime.now().isoformat(),
        scoring_mode="production",
    )


def _humanize_feature(feature_name: str) -> str:
    """Convert raw feature names to human-readable labels."""
    mappings = {
        'TransactionAmt': 'Transaction amount',
        'TransactionAmt_log': 'Transaction amount (log-scaled)',
        'cents': 'Cents portion of amount',
        'ProductCD': 'Product category',
        'card4': 'Card brand',
        'card6': 'Card type',
        'P_emaildomain': 'Purchaser email domain',
        'DeviceType': 'Device type',
        'outsider15': 'D1/D15 identity inconsistency',
        'uid_FE': 'UID frequency (how common this identity is)',
        'TransactionAmt_uid_mean': 'Average transaction amount for this identity',
        'TransactionAmt_uid_std': 'Transaction amount volatility for this identity',
    }
    return mappings.get(feature_name, feature_name)


def _build_heuristic_factors(txn: TransactionInput, score: float) -> list:
    """Build risk factor explanations from UI inputs when SHAP is unavailable."""
    factors = []
    if txn.transaction_amount > 500:
        factors.append({"feature": "High transaction amount", "impact": "high", "direction": "increases_risk"})
    if txn.product_cd in ('C', 'S'):
        factors.append({"feature": f"Product code '{txn.product_cd}' is higher-risk", "impact": "high", "direction": "increases_risk"})
    if txn.email_domain.lower() in ('mail.com', 'outlook.com'):
        factors.append({"feature": f"Email domain '{txn.email_domain}' has elevated fraud rate", "impact": "high", "direction": "increases_risk"})
    if txn.device_type.lower() == 'mobile':
        factors.append({"feature": "Mobile device (higher fraud rate)", "impact": "medium", "direction": "increases_risk"})
    if txn.is_international:
        factors.append({"feature": "International transaction", "impact": "medium", "direction": "increases_risk"})
    if score < 0.1:
        factors.insert(0, {"feature": "Standard transaction profile", "impact": "low", "direction": "decreases_risk"})
    return factors[:5]


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
        scored_at=datetime.now().isoformat(),
        scoring_mode="demo",
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
        "features_loaded": feature_list is not None,
        "encoders_loaded": label_encoders is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/model/info")
async def model_info():
    if model_loaded and scorer:
        info = scorer.get_model_info()
        info['scoring_mode'] = 'production'
        return info
    return {
        "model_type": "XGBoost + LightGBM Ensemble (Demo Mode)",
        "scoring_mode": "demo",
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
    """
    Score a single merchant transaction.

    Forces the use of heuristic demo scoring for the UI because the real
    model expects 434 complex historical features, not just 9 simple UI fields.
    """
    return demo_score(txn)


@app.post("/api/score/batch")
async def score_batch(batch: BatchInput):
    """Score multiple transactions."""
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
            "scoring_mode": "production" if model_loaded else "demo",
        }
    }


@app.post("/api/score/csv")
async def score_csv(file: UploadFile = File(...)):
    """Score a batch of transactions uploaded as a CSV file using the ML model."""
    if not model_loaded or not scorer:
        raise HTTPException(status_code=503, detail="ML Model is not loaded. Cannot process full CSVs.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
    
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV is empty")
    
    # Process through model
    try:
        res = scorer.score(df, explain=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model scoring failed: {e}")
    
    scores = res['risk_scores']
    tiers = res['risk_tiers']
    
    # Build summary
    tier_counts = {}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
        
    # Return top 100 for table display
    display_df = df.copy()
    display_df['risk_score'] = scores
    display_df['risk_tier'] = tiers
    
    # Sort by risk score (highest first) and take top 100
    display_df = display_df.sort_values('risk_score', ascending=False).head(100)
    
    # Select important columns to send back
    cols_to_return = []
    if 'TransactionID' in display_df.columns: cols_to_return.append('TransactionID')
    if 'TransactionAmt' in display_df.columns: cols_to_return.append('TransactionAmt')
    if 'ProductCD' in display_df.columns: cols_to_return.append('ProductCD')
    if 'P_emaildomain' in display_df.columns: cols_to_return.append('P_emaildomain')
    if 'card4' in display_df.columns: cols_to_return.append('card4')
    
    # Also include the score and tier
    results = []
    for _, row in display_df.iterrows():
        r = {c: row[c] for c in cols_to_return}
        r['risk_score'] = round(row['risk_score'], 4)
        r['risk_tier'] = row['risk_tier']
        results.append(r)
        
    return {
        "results": results,
        "summary": {
            "total": len(df),
            "mean_score": round(np.mean(scores), 4),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "tier_distribution": tier_counts,
            "scoring_mode": "production"
        }
    }


@app.post("/api/generate-alert")
async def generate_alert(req: AlertRequest):
    """
    Generate a professional email alert using an LLM based on ML risk factors.
    Acts as the 'Auto-Responder' required by Track 02.
    """
    # Create the prompt based on the ML explanation
    factors_str = "\\n".join([f"- {f.get('feature')}: {f.get('description')} (Value: {f.get('value')})" for f in req.risk_factors])
    
    prompt = f"""
    You are an AI Risk Manager at Razorpay. Write a professional, concise email to a merchant whose payout has been temporarily held due to a {req.risk_tier} risk transaction.

    Merchant Name: {req.merchant_name}
    Transaction ID: {req.transaction_id}
    Amount: ${req.amount}
    
    The ML model flagged this transaction due to the following risk factors:
    {factors_str}

    The email should:
    1. Be polite but firm.
    2. Explain that the transaction triggered automated security systems.
    3. Briefly allude to the risk factors without giving away exact ML parameters.
    4. Request specific verification documents (like an invoice or shipping proof).
    """

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return {"email_draft": response.text}
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Fallback to template if API fails
            pass

    # Fallback Template if no API key or API fails
    fallback_email = f"""Subject: Action Required: Verification needed for Transaction {req.transaction_id}

Dear {req.merchant_name},

Our automated security systems have placed a temporary hold on the payout for Transaction {req.transaction_id} (${req.amount}) as it was flagged as {req.risk_tier} risk. 

Based on our analysis, there were irregular patterns detected related to the transaction velocity and device metrics associated with this charge.

To protect both you and your customers from potential chargebacks or fraudulent activity, we require additional verification before releasing these funds. 

Please reply to this email with:
1. A copy of the customer invoice.
2. Proof of shipping or delivery.

Once verified, the hold will be released immediately.

Best regards,
Razorpay Risk Team"""

    return {"email_draft": fallback_email}


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
