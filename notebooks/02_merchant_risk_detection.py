# %% [markdown]
# # 🛡️ Post-Onboarding Merchant Risk & Fraud Detection
# # Transaction Monitoring Engine
# 
# ## RazorPay AI Buildathon — AI Risk Manager Track
# 
# **Problem Statement**: Merchants who pass KYC checks can still engage in fraudulent 
# processing once active. This pipeline monitors real-time transaction streams during a 
# merchant's early lifecycle — evaluating transaction amounts, card usage patterns, device 
# fingerprints, and temporal signals — to aggregate a merchant-level risk score and trigger 
# mitigations (payout holds, account reviews, account freezes) before financial loss occurs.
#
# ### Architecture
# ```
# ┌─────────────────────────────────────────────────────────────────────┐
# │           EARLY-LIFECYCLE MERCHANT RISK ENGINE                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  Merchant passes KYC → Onboarded → Starts processing transactions  │
# │                                                                     │
# │  1. TRANSACTION STREAM    2. BEHAVIORAL PROFILING   3. RISK SCORING │
# │     • Real-time txns         • Velocity features       • XGBoost    │
# │     • Card/device data       • Amount volatility       • LightGBM   │
# │     • Identity signals       • Card diversity          • Ensemble   │
# │                              • Device entropy                       │
# │                              • Temporal patterns     4. ACTION      │
# │                                                        • Release ✅ │
# │                                                        • Hold 🟡    │
# │                                                        • Review 🟠  │
# │                                                        • Freeze 🔴  │
# └─────────────────────────────────────────────────────────────────────┘
# ```
#
# ### Dataset
# [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — Real 
# anonymised transaction data from **Vesta Corporation**, a leading payment processor.
# - 590,540 training transactions | 506,691 test transactions
# - 394 transaction features + 41 identity features  
# - **3.50% fraud rate** (extreme class imbalance — mirrors real-world merchant risk)

# %% [markdown]
# ---
# ## 📦 Phase 0: Environment Setup

# %%
# ============================================================================
# IMPORTS
# ============================================================================
import sys
import os
import gc
import time
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve
)

import xgboost as xgb
import lightgbm as lgb
import shap
from tqdm import tqdm

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if 'notebooks' in os.getcwd() else os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_ieee_data, print_system_info, get_output_dir, get_model_dir
from src.feature_engineering import (
    run_feature_engineering, select_features, save_feature_list
)
from src.model_utils import (
    save_xgb_checkpoint, save_lgb_checkpoint,
    save_final_models, build_ensemble_config, build_training_metadata,
    generate_model_card
)

# Colour palette
COLORS = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'success': '#059669',
    'danger': '#DC2626',
    'warning': '#D97706',
    'bg_dark': '#0F172A',
    'bg_card': '#1E293B',
    'text': '#F1F5F9',
    'fraud': '#EF4444',
    'legit': '#22C55E',
    'accent1': '#06B6D4',
    'grid': '#334155',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg_dark'],
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': '#475569',
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'legend.facecolor': COLORS['bg_card'],
    'legend.edgecolor': '#475569',
    'font.size': 11,
})

# Output directories
EDA_DIR = get_output_dir('eda')
EVAL_DIR = get_output_dir('model_evaluation')
SHAP_DIR = get_output_dir('shap')
CHECKPOINT_DIR = get_model_dir('checkpoints')
FINAL_DIR = get_model_dir('final')
ARTIFACT_DIR = get_model_dir('artifacts')

print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_system_info()
print(f"\n📊 XGBoost: {xgb.__version__} | LightGBM: {lgb.__version__}")

# %% [markdown]
# ---
# ## 📊 Phase 1: Data Ingestion

# %%
data = load_ieee_data(load_test=True, verbose=True)
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']

# %% [markdown]
# ---
# ## 🔍 Phase 2: EDA Highlights
# 
# Full EDA is in `notebooks/01_eda.ipynb`. Here we show the key validated plots 
# that inform our feature engineering decisions.
# 
# **Key EDA Findings:**
# - 3.50% fraud rate → extreme 28:1 class imbalance
# - ProductCD 'C' has 11.7% fraud rate (3x average) — high-risk product category
# - Discover cards have 7.7% fraud rate vs visa at 3.5%
# - Credit cards: 6.7% fraud vs debit: 2.4% — credit fraud is 2.8x more prevalent
# - Fraud peaks at hours 4-8 (early morning) at ~10% vs ~2.5% during business hours
# - mail.com, outlook.com are highest-risk email domains (19%, 9.5%)
# - Mobile devices show 10.2% fraud rate vs desktop 6.5%
# - V257 has strongest correlation with fraud (|r|=0.383)

# %%
# Display key EDA plots from outputs/eda/
from IPython.display import Image, display

eda_plots = ['class_distribution.png', 'product_card_analysis.png', 
             'temporal_patterns.png', 'email_domain_analysis.png']

for plot_name in eda_plots:
    plot_path = os.path.join(EDA_DIR, plot_name)
    if os.path.exists(plot_path):
        print(f"📊 {plot_name}")
        # In script mode, just confirm the plot exists
        print(f"   ✅ Found at: {plot_path}")

# %% [markdown]
# ---
# ## ⚙️ Phase 3: Feature Engineering
# 
# This is the **most critical phase**. We engineer 47+ features that capture 
# merchant-level transaction behaviour patterns:
# 1. **Magic UID**: Pseudo-merchant identity (card1 + addr1 + D1)
# 2. **Transaction Velocity**: How fast is money moving?
# 3. **Amount Statistics**: Mean, std, deviation from merchant average
# 4. **Device & Email Diversity**: High diversity = risk signal
# 5. **Temporal Patterns**: Hour-of-day transaction behaviour

# %%
X_train, X_test, artifacts = run_feature_engineering(
    X_train, X_test, y_train, save_dir=ARTIFACT_DIR
)

# %%
# Feature selection
cols = select_features(X_train)

# Ensure cols only contains features present in BOTH train and test
# (identity table has different coverage — some id_* columns may be missing from X_test)
missing_in_test = [c for c in cols if c not in X_test.columns]
if missing_in_test:
    print(f"   ⚠️  Dropping {len(missing_in_test)} features missing from X_test: {missing_in_test}")
    cols = [c for c in cols if c in X_test.columns]
    print(f"   ✅ Adjusted feature count: {len(cols)}")

save_feature_list(cols, ARTIFACT_DIR)

# %% [markdown]
# ---
# ## 🌲 Phase 5: Model Training — XGBoost (Primary Model)
# 
# **Cross-Validation**: GroupKFold by month (temporal split — prevents data leakage)
# 
# **Key**: Every fold model is checkpointed to `models/checkpoints/`. If training 
# crashes at fold 5, we have folds 1-4 saved and can resume.

# %%
# ============================================================================
# XGBOOST TRAINING WITH CHECKPOINTING
# ============================================================================
print("=" * 70)
print("  MODEL TRAINING: XGBoost (with fold checkpointing)")
print("=" * 70)

n_fraud = y_train.sum()
n_legit = len(y_train) - n_fraud
scale_pos = n_legit / n_fraud
print(f"\n⚖️  Class imbalance ratio: 1:{scale_pos:.0f}")
print(f"   Using scale_pos_weight = {scale_pos:.1f}")

xgb_params = {
    'n_estimators': 2500,
    'max_depth': 12,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'missing': -1,
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
}

oof_xgb = np.zeros(len(X_train))
preds_xgb = np.zeros(len(X_test))

skf = GroupKFold(n_splits=6)
fold_scores_xgb = []

t_xgb_start = time.time()

pbar_xgb = tqdm(skf.split(X_train, y_train, groups=X_train['DT_M']),
                total=6, desc='🌲 XGBoost Training', unit='fold',
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

for fold_i, (idxT, idxV) in enumerate(pbar_xgb):
    month = X_train.iloc[idxV]['DT_M'].iloc[0]
    pbar_xgb.set_postfix({'fold': fold_i+1, 'month': month})
    print(f'\n{"─" * 50}')
    print(f'  📁 Fold {fold_i + 1}/6 — Withholding Month {month}')
    print(f'     Train: {len(idxT):,} rows | Validation: {len(idxV):,} rows')
    print(f'{"─" * 50}')
    
    clf = xgb.XGBClassifier(**xgb_params)
    
    clf.fit(
        X_train[cols].iloc[idxT], y_train.iloc[idxT],
        eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
        verbose=200,
    )
    
    oof_xgb[idxV] = clf.predict_proba(X_train[cols].iloc[idxV])[:, 1]
    preds_xgb += clf.predict_proba(X_test[cols])[:, 1] / skf.n_splits
    
    fold_auc = roc_auc_score(y_train.iloc[idxV], oof_xgb[idxV])
    fold_scores_xgb.append(fold_auc)
    pbar_xgb.set_postfix({'fold': fold_i+1, 'AUC': f'{fold_auc:.4f}'})
    print(f'  🎯 Fold {fold_i + 1} AUC: {fold_auc:.6f}')
    
    # *** CHECKPOINT: Save fold model ***
    save_xgb_checkpoint(clf, fold_i + 1, oof_xgb[idxV], idxV, CHECKPOINT_DIR)
    
    del clf
    gc.collect()

overall_auc_xgb = roc_auc_score(y_train, oof_xgb)
xgb_train_time = time.time() - t_xgb_start

print(f'\n{"=" * 70}')
print(f'  📊 XGBoost RESULTS')
print(f'{"=" * 70}')
print(f'  Overall OOF AUC:     {overall_auc_xgb:.6f}')
print(f'  Mean Fold AUC:       {np.mean(fold_scores_xgb):.6f} ± {np.std(fold_scores_xgb):.6f}')
print(f'  Training Time:       {xgb_train_time/60:.1f} minutes')
print(f'{"=" * 70}')

X_train['oof_xgb'] = oof_xgb

# %% [markdown]
# ---
# ## 🌿 Phase 6: Model Training — LightGBM (Secondary Model)

# %%
# ============================================================================
# LIGHTGBM TRAINING WITH CHECKPOINTING
# ============================================================================
print("=" * 70)
print("  MODEL TRAINING: LightGBM (with fold checkpointing)")
print("=" * 70)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 256,
    'max_depth': 12,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'scale_pos_weight': scale_pos,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1,
}

oof_lgb = np.zeros(len(X_train))
preds_lgb = np.zeros(len(X_test))
fold_scores_lgb = []

t_lgb_start = time.time()

pbar_lgb = tqdm(skf.split(X_train, y_train, groups=X_train['DT_M']),
                total=6, desc='🌿 LightGBM Training', unit='fold',
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

for fold_i, (idxT, idxV) in enumerate(pbar_lgb):
    month = X_train.iloc[idxV]['DT_M'].iloc[0]
    pbar_lgb.set_postfix({'fold': fold_i+1, 'month': month})
    print(f'\n{"─" * 50}')
    print(f'  📁 Fold {fold_i + 1}/6 — Withholding Month {month}')
    print(f'     Train: {len(idxT):,} rows | Validation: {len(idxV):,} rows')
    print(f'{"─" * 50}')
    
    lgb_train = lgb.Dataset(X_train[cols].iloc[idxT], y_train.iloc[idxT])
    lgb_val   = lgb.Dataset(X_train[cols].iloc[idxV], y_train.iloc[idxV], reference=lgb_train)
    
    callbacks = [
        lgb.log_evaluation(period=200),
        lgb.early_stopping(stopping_rounds=200),
    ]
    
    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2500,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )
    
    oof_lgb[idxV] = model_lgb.predict(X_train[cols].iloc[idxV])
    preds_lgb += model_lgb.predict(X_test[cols]) / skf.n_splits
    
    fold_auc_lgb = roc_auc_score(y_train.iloc[idxV], oof_lgb[idxV])
    fold_scores_lgb.append(fold_auc_lgb)
    pbar_lgb.set_postfix({'fold': fold_i+1, 'AUC': f'{fold_auc_lgb:.4f}'})
    print(f'  🎯 Fold {fold_i + 1} AUC: {fold_auc_lgb:.6f}')
    
    # *** CHECKPOINT: Save fold model ***
    save_lgb_checkpoint(model_lgb, fold_i + 1, oof_lgb[idxV], idxV, CHECKPOINT_DIR)
    
    del model_lgb, lgb_train, lgb_val
    gc.collect()

overall_auc_lgb = roc_auc_score(y_train, oof_lgb)
lgb_train_time = time.time() - t_lgb_start

print(f'\n{"=" * 70}')
print(f'  📊 LightGBM RESULTS')
print(f'{"=" * 70}')
print(f'  Overall OOF AUC:     {overall_auc_lgb:.6f}')
print(f'  Mean Fold AUC:       {np.mean(fold_scores_lgb):.6f} ± {np.std(fold_scores_lgb):.6f}')
print(f'  Training Time:       {lgb_train_time/60:.1f} minutes')
print(f'{"=" * 70}')

X_train['oof_lgb'] = oof_lgb

# %% [markdown]
# ---
# ## 🤝 Phase 7: Ensemble & Risk Score Generation

# %%
# ============================================================================
# ENSEMBLE: WEIGHTED BLEND
# ============================================================================
print("=" * 70)
print("  ENSEMBLE: XGBoost + LightGBM Blend")
print("=" * 70)

best_auc = 0
best_w = 0
for w in np.arange(0.0, 1.01, 0.05):
    blended = w * oof_xgb + (1 - w) * oof_lgb
    auc = roc_auc_score(y_train, blended)
    if auc > best_auc:
        best_auc = auc
        best_w = w

print(f"\n  🏆 Optimal XGBoost weight: {best_w:.2f}")
print(f"  🏆 Optimal LightGBM weight: {1-best_w:.2f}")
print(f"  🏆 Blended OOF AUC: {best_auc:.6f}")
print(f"\n  📈 Improvement over XGBoost alone: {(best_auc - overall_auc_xgb)*10000:.1f} bps")
print(f"  📈 Improvement over LightGBM alone: {(best_auc - overall_auc_lgb)*10000:.1f} bps")

final_oof = best_w * oof_xgb + (1 - best_w) * oof_lgb
final_preds = best_w * preds_xgb + (1 - best_w) * preds_lgb

# %% [markdown]
# ---
# ## 📊 Phase 8: Model Evaluation & Risk Analysis

# %%
# ============================================================================
# PERFORMANCE DASHBOARD
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🛡️ Post-Onboarding Merchant Risk — Model Performance Dashboard',
             fontsize=18, fontweight='bold', y=0.98)

# --- ROC Curve ---
ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_train, final_oof)
ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
        label=f'Ensemble (AUC = {best_auc:.4f})')
fpr_xgb, tpr_xgb, _ = roc_curve(y_train, oof_xgb)
ax.plot(fpr_xgb, tpr_xgb, color=COLORS['secondary'], linewidth=1.5, alpha=0.7,
        label=f'XGBoost (AUC = {overall_auc_xgb:.4f})')
fpr_lgb, tpr_lgb, _ = roc_curve(y_train, oof_lgb)
ax.plot(fpr_lgb, tpr_lgb, color=COLORS['success'], linewidth=1.5, alpha=0.7,
        label=f'LightGBM (AUC = {overall_auc_lgb:.4f})')
ax.plot([0, 1], [0, 1], 'w--', alpha=0.3)
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(labelcolor=COLORS['text'], fontsize=10)

# --- PR Curve ---
ax = axes[0, 1]
precision, recall, _ = precision_recall_curve(y_train, final_oof)
pr_auc = average_precision_score(y_train, final_oof)
ax.plot(recall, precision, color=COLORS['fraud'], linewidth=2.5,
        label=f'PR AUC = {pr_auc:.4f}')
ax.axhline(y=y_train.mean(), color=COLORS['warning'], linestyle='--', alpha=0.7,
           label=f'Baseline = {y_train.mean():.4f}')
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.set_xlabel('Recall (Fraud Caught)')
ax.set_ylabel('Precision')
ax.legend(labelcolor=COLORS['text'], fontsize=10)

# --- Risk Score Distribution ---
ax = axes[1, 0]
ax.hist(final_oof[y_train == 0], bins=100, alpha=0.7, color=COLORS['legit'],
        label='Legitimate', density=True)
ax.hist(final_oof[y_train == 1], bins=100, alpha=0.7, color=COLORS['fraud'],
        label='Fraud', density=True)
ax.set_title('Risk Score Distribution', fontweight='bold')
ax.set_xlabel('Risk Score')
ax.set_ylabel('Density')
ax.legend(labelcolor=COLORS['text'], fontsize=10)

# --- Fold AUC Comparison ---
ax = axes[1, 1]
x_pos = np.arange(len(fold_scores_xgb))
width = 0.35
ax.bar(x_pos - width/2, fold_scores_xgb, width, label='XGBoost',
       color=COLORS['primary'], edgecolor='white', linewidth=0.5)
ax.bar(x_pos + width/2, fold_scores_lgb, width, label='LightGBM',
       color=COLORS['success'], edgecolor='white', linewidth=0.5)
ax.set_title('AUC per Fold (GroupKFold by Month)', fontweight='bold')
ax.set_xlabel('Fold')
ax.set_ylabel('AUC Score')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_scores_xgb))])
ax.legend(labelcolor=COLORS['text'], fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(EVAL_DIR, 'performance_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/model_evaluation/performance_dashboard.png")

# %%
# ============================================================================
# CONFUSION MATRIX
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (thresh, title) in enumerate([
    (0.5, 'Threshold = 0.5 (Standard)'),
    (0.2, 'Threshold = 0.2 (High Sensitivity)')
]):
    ax = axes[i]
    preds_binary = (final_oof >= thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_title(f'Confusion Matrix — {title}', fontweight='bold', pad=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'),
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/model_evaluation/confusion_matrix.png")

# %%
# ============================================================================
# OPERATIONAL THRESHOLD ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("  🎚️  OPERATIONAL THRESHOLD ANALYSIS")
print("=" * 70)
print("\n  Scenario: Post-onboarding merchant transaction monitoring\n")

thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
print(f"  {'Threshold':>10} │ {'Precision':>10} │ {'Recall':>10} │ {'F1':>10} │ {'Flagged%':>10} │ {'Action'}")
print(f"  {'─'*10} │ {'─'*10} │ {'─'*10} │ {'─'*10} │ {'─'*10} │ {'─'*30}")

for thresh in thresholds:
    preds_binary = (final_oof >= thresh).astype(int)
    prec = preds_binary[y_train == 1].sum() / max(preds_binary.sum(), 1)
    rec  = preds_binary[y_train == 1].sum() / max(y_train.sum(), 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-10)
    flagged = preds_binary.mean() * 100
    
    if thresh <= 0.05:
        action = "🟢 Release payouts"
    elif thresh <= 0.2:
        action = "🟡 Enhanced monitoring"
    elif thresh <= 0.5:
        action = "🟠 Hold payouts — manual review"
    else:
        action = "🔴 Freeze account"
    
    print(f"  {thresh:>10.2f} │ {prec:>10.4f} │ {rec:>10.4f} │ {f1:>10.4f} │ {flagged:>9.2f}% │ {action}")

# %% [markdown]
# ---
# ## 🔍 Phase 9: SHAP Explainability
# 
# Every risk decision must be **explainable** — RBI regulatory requirements 
# demand transparent risk decisions. SHAP provides mathematically rigorous 
# feature attribution for each prediction.

# %%
# ============================================================================
# SHAP ANALYSIS — Train dedicated model for SHAP
# ============================================================================
print("=" * 70)
print("  SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

print("\n🌲 Training XGBoost model for SHAP analysis...")

# Use 75/25 split for SHAP computation
idxT_shap = X_train.index[:3 * len(X_train) // 4]
idxV_shap = X_train.index[3 * len(X_train) // 4:]

clf_shap = xgb.XGBClassifier(
    n_estimators=1500, max_depth=12, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.4, missing=-1,
    eval_metric='auc', scale_pos_weight=scale_pos,
    tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
)

clf_shap.fit(
    X_train.loc[idxT_shap, cols], y_train.loc[idxT_shap],
    eval_set=[(X_train.loc[idxV_shap, cols], y_train.loc[idxV_shap])],
    verbose=500,
)

shap_auc = roc_auc_score(y_train[idxV_shap], clf_shap.predict_proba(X_train.loc[idxV_shap, cols])[:, 1])
print(f"\n   SHAP model validation AUC: {shap_auc:.6f}")

# %%
print("\n📊 Computing SHAP values on sample of 5,000 transactions...")
sample_idx = np.random.choice(idxV_shap, size=min(5000, len(idxV_shap)), replace=False)
X_sample = X_train.loc[sample_idx, cols]

explainer = shap.TreeExplainer(clf_shap)
shap_values = explainer.shap_values(X_sample)
print(f"   ✅ SHAP values computed: {shap_values.shape}")

# %%
# --- SHAP Feature Importance ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=25, show=False)
plt.title('🔍 Top 25 Risk Drivers — SHAP Feature Importance',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, 'feature_importance.png'),
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/shap/feature_importance.png")

# %%
# --- SHAP Beeswarm ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, max_display=25, show=False)
plt.title('🐝 SHAP Beeswarm — Feature Impact on Risk Score',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, 'beeswarm.png'),
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/shap/beeswarm.png")

# %%
# --- SHAP Waterfall: Explain a High-Risk Transaction ---
print("\n🔍 Explaining a single HIGH-RISK merchant transaction...")
high_risk_idx = sample_idx[np.argsort(clf_shap.predict_proba(X_sample)[:, 1])[-1]]
high_risk_score = clf_shap.predict_proba(X_train.loc[[high_risk_idx], cols])[:, 1][0]
actual_label = y_train[high_risk_idx]

print(f"   Transaction ID: {high_risk_idx}")
print(f"   Risk Score: {high_risk_score:.4f}")
print(f"   Actual: {'🔴 FRAUD' if actual_label == 1 else '🟢 LEGITIMATE'}")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
shap_idx = np.where(sample_idx == high_risk_idx)[0][0]
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[shap_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[shap_idx],
        feature_names=cols
    ),
    max_display=15, show=False
)
plt.title(f'🔍 Why Was This Merchant Flagged?\n(Risk: {high_risk_score:.4f} | Actual: {"FRAUD" if actual_label else "LEGIT"})',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, 'waterfall_highrisk.png'),
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/shap/waterfall_highrisk.png")

# %% [markdown]
# ---
# ## 💾 Phase 10: Save Final Production Models
# 
# Retrain final models on ALL training data (CV folds were for evaluation only).

# %%
# ============================================================================
# RETRAIN FINAL MODELS ON ALL DATA
# ============================================================================
print("=" * 70)
print("  TRAINING FINAL PRODUCTION MODELS")
print("=" * 70)

# --- Final XGBoost ---
print("\n🌲 Training final XGBoost on all training data...")
xgb_final = xgb.XGBClassifier(**xgb_params)
xgb_final.fit(X_train[cols], y_train, verbose=500)
print("   ✅ XGBoost final model trained")

# --- Final LightGBM ---
print("\n🌿 Training final LightGBM on all training data...")
lgb_train_full = lgb.Dataset(X_train[cols], y_train)
lgb_final = lgb.train(
    lgb_params,
    lgb_train_full,
    num_boost_round=2500,
    callbacks=[lgb.log_evaluation(period=500)],
)
print("   ✅ LightGBM final model trained")

# %%
# ============================================================================
# SAVE EVERYTHING
# ============================================================================
print("\n💾 Saving all models and metadata...")

ensemble_config = build_ensemble_config(
    xgb_weight=best_w, lgb_weight=1-best_w,
    ensemble_auc=best_auc,
    xgb_auc=overall_auc_xgb, lgb_auc=overall_auc_lgb,
    feature_count=len(cols)
)

training_metadata = build_training_metadata(
    xgb_params=xgb_params, lgb_params=lgb_params,
    fold_scores_xgb=fold_scores_xgb, fold_scores_lgb=fold_scores_lgb,
    ensemble_auc=best_auc,
    training_rows=len(X_train), feature_count=len(cols),
    xgb_train_time=xgb_train_time, lgb_train_time=lgb_train_time
)

save_final_models(xgb_final, lgb_final, ensemble_config, training_metadata, FINAL_DIR)
generate_model_card(training_metadata, ensemble_config, FINAL_DIR)

# Also save SHAP model (used by inference for explanations)
clf_shap.save_model(os.path.join(FINAL_DIR, 'xgb_shap.json'))
print(f"   💾 SHAP model saved: {FINAL_DIR}/xgb_shap.json")

del clf_shap, explainer, shap_values, X_sample
gc.collect()

# %% [markdown]
# ---
# ## 📝 Phase 11: Submission & Risk Tiers

# %%
# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("=" * 70)
print("  GENERATING FINAL OUTPUTS")
print("=" * 70)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'ieee-fraud-detection')
sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sample_submission['isFraud'] = final_preds
submissions_dir = os.path.join(PROJECT_ROOT, 'submissions')
os.makedirs(submissions_dir, exist_ok=True)
sample_submission.to_csv(os.path.join(submissions_dir, 'submission_ensemble.csv'), index=False)
print(f"\n✅ Submission saved: submissions/submission_ensemble.csv")
print(f"   Predictions: {len(final_preds):,} transactions scored")
print(f"   Mean risk score: {final_preds.mean():.6f}")

# %%
# ============================================================================
# RISK TIER CLASSIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("  🏷️  MERCHANT RISK TIER SUMMARY")
print("=" * 70)

risk_tiers = pd.cut(final_preds, 
                     bins=[0, 0.05, 0.2, 0.5, 1.0],
                     labels=['🟢 LOW RISK — Release Payouts', 
                             '🟡 MEDIUM RISK — Enhanced Monitoring',
                             '🟠 HIGH RISK — Hold Payouts', 
                             '🔴 CRITICAL RISK — Freeze Account'])

tier_counts = risk_tiers.value_counts()
print(f"\n  Risk Tier Distribution (Test Set):\n")
for tier, count in tier_counts.items():
    pct = count / len(final_preds) * 100
    print(f"  {tier}: {count:>8,} merchants ({pct:5.1f}%)")

# %%
# ============================================================================
# FINAL RISK SCORE DISTRIBUTION
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.hist(final_preds, bins=200, color=COLORS['primary'], edgecolor='none', alpha=0.8)
ax.axvline(x=0.05, color=COLORS['legit'], linestyle='--', linewidth=2, alpha=0.8, 
           label='Release Threshold (0.05)')
ax.axvline(x=0.2,  color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.8, 
           label='Hold Threshold (0.20)')
ax.axvline(x=0.5,  color=COLORS['fraud'], linestyle='--', linewidth=2, alpha=0.8, 
           label='Freeze Threshold (0.50)')

ax.set_title('🛡️ Merchant Risk Score Distribution — Post-Onboarding Monitoring\nWith Operational Decision Thresholds',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Risk Score (0 = Safe → 1 = Critical)', fontsize=12)
ax.set_ylabel('Number of Merchants', fontsize=12)
ax.legend(labelcolor=COLORS['text'], fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'risk_score_distribution.png'),
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/model_evaluation/risk_score_distribution.png")

# %% [markdown]
# ---
# ## 🎭 Phase 12: Simulated Merchant Risk Assessment
# 
# Demonstrate the system with realistic Indian merchant profiles to show how 
# the model would score merchants in a real Razorpay deployment.

# %%
# ============================================================================
# SIMULATED MERCHANT PROFILES
# ============================================================================
print("=" * 70)
print("  🎭 SIMULATED MERCHANT RISK ASSESSMENT")
print("=" * 70)

# We'll use real transactions from the test set as proxy merchant profiles
# Group by risk tier and show representative examples

print("\n📊 Sample Merchant Profiles from Scored Test Set:\n")

# Create merchant-level profiles by sampling from different risk tiers
tier_samples = {
    '🟢 Sharma Electronics, Jaipur — Mid-scale retailer': 
        (final_preds < 0.05),
    '🟡 QuickPay Digital Services, Mumbai — Online payments': 
        ((final_preds >= 0.05) & (final_preds < 0.2)),
    '🟠 FastCash Transfers, Delhi — Money transfer service': 
        ((final_preds >= 0.2) & (final_preds < 0.5)),
    '🔴 Anonymous Gift Cards Ltd — Prepaid card reseller': 
        (final_preds >= 0.5),
}

for merchant_name, mask in tier_samples.items():
    if mask.sum() > 0:
        # Pick a representative transaction
        idx = np.where(mask)[0]
        representative_score = final_preds[idx[len(idx)//2]]  # median of tier
        
        # Determine action
        if representative_score < 0.05:
            action = "✅ RELEASE PAYOUTS — Low risk, auto-approve"
            tier = "LOW RISK"
        elif representative_score < 0.2:
            action = "🟡 ENHANCED MONITORING — Track closely for 30 days"
            tier = "MEDIUM RISK"
        elif representative_score < 0.5:
            action = "🟠 HOLD PAYOUTS — Manual review required before release"
            tier = "HIGH RISK"
        else:
            action = "🔴 FREEZE ACCOUNT — Immediate investigation required"
            tier = "CRITICAL RISK"
        
        print(f"  ┌─────────────────────────────────────────────────────")
        print(f"  │ {merchant_name}")
        print(f"  │ Risk Score: {representative_score:.4f} → {tier}")
        print(f"  │ Action: {action}")
        print(f"  │ Transactions in tier: {mask.sum():,}")
        print(f"  └─────────────────────────────────────────────────────\n")

# %% [markdown]
# ---
# ## ✅ Summary & MLOps Considerations
# 
# ### What We Built
# An end-to-end **Post-Onboarding Merchant Risk Detection Engine** that:
# 1. **Monitors** merchant transaction streams using 590K+ real transactions from Vesta
# 2. **Engineers 47+ behavioral features** including UID identity, velocity, and device entropy
# 3. **Scores transactions** with an XGBoost + LightGBM ensemble (AUC ~0.95)
# 4. **Explains risk decisions** using SHAP analysis (RBI-compliant transparency)
# 5. **Classifies merchants into 4 action tiers**: Release / Monitor / Hold / Freeze
#
# ### MLOps Considerations (Production Deployment)
# - **CI/CD**: Model retraining triggered via GitHub Actions when new fraud labels arrive
# - **Model Monitoring**: Track prediction drift using Evidently AI; alert if risk score 
#   distribution shifts >5% from baseline
# - **Feature Store**: Aggregation features (uid_mean, uid_std) pre-computed in Feast/Tecton
# - **A/B Testing**: Shadow-score new model versions for 2 weeks before promotion
# - **Data Versioning**: DVC/LakeFS to version training datasets and link models to data snapshots
# - **Model Registry**: MLflow for versioned model storage, lineage tracking, and deployment

# %%
print("\n" + "=" * 70)
print("  🎉 PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n  📁 Models saved to:     models/final/")
print(f"  📁 Checkpoints at:      models/checkpoints/")
print(f"  📁 Artifacts at:        models/artifacts/")
print(f"  📁 Evaluation plots at: outputs/model_evaluation/")
print(f"  📁 SHAP plots at:       outputs/shap/")
print(f"  📁 EDA plots at:        outputs/eda/")
print(f"\n  🛡️  RazorPay AI Buildathon — Post-Onboarding Merchant Risk Detection")
print("=" * 70)
