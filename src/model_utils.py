"""
model_utils.py — Model training utilities with checkpointing.

Handles:
- Saving/loading model checkpoints per fold
- Saving final production models
- Training metadata tracking
- Model card generation
"""

import os
import json
import time
import platform
import numpy as np
from datetime import datetime


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
def save_xgb_checkpoint(model, fold_i, oof_preds, val_indices, checkpoint_dir):
    """Save an XGBoost fold model and its OOF predictions."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoint_dir, f'xgb_fold_{fold_i}.json')
    oof_path = os.path.join(checkpoint_dir, f'oof_xgb_fold_{fold_i}.npy')
    idx_path = os.path.join(checkpoint_dir, f'idx_xgb_fold_{fold_i}.npy')
    
    model.save_model(model_path)
    np.save(oof_path, oof_preds)
    np.save(idx_path, val_indices)
    
    print(f"   💾 Checkpoint saved: xgb_fold_{fold_i}.json "
          f"({os.path.getsize(model_path)/1024/1024:.1f} MB)")


def save_lgb_checkpoint(model, fold_i, oof_preds, val_indices, checkpoint_dir):
    """Save a LightGBM fold model and its OOF predictions."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoint_dir, f'lgb_fold_{fold_i}.txt')
    oof_path = os.path.join(checkpoint_dir, f'oof_lgb_fold_{fold_i}.npy')
    idx_path = os.path.join(checkpoint_dir, f'idx_lgb_fold_{fold_i}.npy')
    
    model.save_model(model_path)
    np.save(oof_path, oof_preds)
    np.save(idx_path, val_indices)
    
    print(f"   💾 Checkpoint saved: lgb_fold_{fold_i}.txt")


# ============================================================================
# FINAL MODEL SAVING
# ============================================================================
def save_final_models(xgb_model, lgb_model, ensemble_config, training_metadata, 
                      model_dir, feature_list=None):
    """
    Save production-ready models with all metadata.
    
    Saves:
    - xgb_final.json — XGBoost model
    - lgb_final.txt — LightGBM model  
    - ensemble_config.json — blend weights, thresholds
    - training_metadata.json — AUCs, training time, params, date
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save XGBoost
    xgb_path = os.path.join(model_dir, 'xgb_final.json')
    xgb_model.save_model(xgb_path)
    print(f"   💾 XGBoost model: {xgb_path} ({os.path.getsize(xgb_path)/1024/1024:.1f} MB)")
    
    # Save LightGBM
    lgb_path = os.path.join(model_dir, 'lgb_final.txt')
    lgb_model.save_model(lgb_path)
    print(f"   💾 LightGBM model: {lgb_path}")
    
    # Save ensemble config
    config_path = os.path.join(model_dir, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2, default=str)
    print(f"   💾 Ensemble config: {config_path}")
    
    # Save training metadata
    meta_path = os.path.join(model_dir, 'training_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(training_metadata, f, indent=2, default=str)
    print(f"   💾 Training metadata: {meta_path}")


def build_ensemble_config(xgb_weight, lgb_weight, ensemble_auc, 
                          xgb_auc, lgb_auc, feature_count):
    """Build the ensemble configuration dictionary."""
    return {
        'xgb_weight': float(xgb_weight),
        'lgb_weight': float(lgb_weight),
        'risk_thresholds': {
            'auto_approve': 0.05,
            'enhanced_review': 0.20,
            'manual_review': 0.50,
            'auto_reject': 1.00
        },
        'risk_tier_labels': {
            'auto_approve': '🟢 LOW RISK — Release Payouts',
            'enhanced_review': '🟡 MEDIUM RISK — Enhanced Monitoring',
            'manual_review': '🟠 HIGH RISK — Hold Payouts for Review',
            'auto_reject': '🔴 CRITICAL RISK — Freeze Account'
        },
        'ensemble_auc': float(ensemble_auc),
        'xgb_auc': float(xgb_auc),
        'lgb_auc': float(lgb_auc),
        'feature_count': int(feature_count),
        'created_at': datetime.now().isoformat()
    }


def build_training_metadata(xgb_params, lgb_params, fold_scores_xgb, fold_scores_lgb,
                            ensemble_auc, training_rows, feature_count,
                            xgb_train_time, lgb_train_time):
    """Build comprehensive training metadata for reproducibility."""
    return {
        'model_type': 'XGBoost + LightGBM Ensemble',
        'problem_type': 'Post-Onboarding Merchant Risk & Fraud Detection',
        'dataset': 'IEEE-CIS Fraud Detection (Vesta Corporation)',
        'training_rows': int(training_rows),
        'feature_count': int(feature_count),
        'cross_validation': {
            'strategy': 'GroupKFold by month',
            'n_splits': len(fold_scores_xgb),
            'group_key': 'DT_M (approximate month)'
        },
        'xgboost': {
            'params': {k: str(v) for k, v in xgb_params.items()},
            'fold_aucs': [float(s) for s in fold_scores_xgb],
            'mean_auc': float(np.mean(fold_scores_xgb)),
            'std_auc': float(np.std(fold_scores_xgb)),
            'training_time_minutes': float(xgb_train_time / 60)
        },
        'lightgbm': {
            'params': {k: str(v) for k, v in lgb_params.items()},
            'fold_aucs': [float(s) for s in fold_scores_lgb],
            'mean_auc': float(np.mean(fold_scores_lgb)),
            'std_auc': float(np.std(fold_scores_lgb)),
            'training_time_minutes': float(lgb_train_time / 60)
        },
        'ensemble_auc': float(ensemble_auc),
        'trained_at': datetime.now().isoformat(),
        'system': {
            'machine': platform.machine(),
            'system': platform.system(),
            'python': platform.python_version()
        }
    }


# ============================================================================
# MODEL CARD GENERATION
# ============================================================================
def generate_model_card(training_metadata, ensemble_config, model_dir):
    """Auto-generate a model card markdown file."""
    meta = training_metadata
    config = ensemble_config
    
    card = f"""# 🛡️ Model Card — Post-Onboarding Merchant Risk Scorer

## Model Details

| Property | Value |
|---|---|
| **Model Type** | {meta['model_type']} |
| **Problem** | {meta['problem_type']} |
| **Dataset** | {meta['dataset']} |
| **Training Rows** | {meta['training_rows']:,} |
| **Features** | {meta['feature_count']} |
| **Trained At** | {meta['trained_at']} |
| **System** | {meta['system']['machine']} — {meta['system']['system']} |

## Performance

| Metric | Value |
|---|---|
| **Ensemble AUC-ROC** | {meta['ensemble_auc']:.4f} |
| **XGBoost Mean AUC** | {meta['xgboost']['mean_auc']:.4f} ± {meta['xgboost']['std_auc']:.4f} |
| **LightGBM Mean AUC** | {meta['lightgbm']['mean_auc']:.4f} ± {meta['lightgbm']['std_auc']:.4f} |
| **XGBoost Weight** | {config['xgb_weight']:.2f} |
| **LightGBM Weight** | {config['lgb_weight']:.2f} |

## Cross-Validation

- **Strategy**: {meta['cross_validation']['strategy']}
- **Folds**: {meta['cross_validation']['n_splits']}
- **Group Key**: {meta['cross_validation']['group_key']}

### XGBoost Fold AUCs
{chr(10).join(f'- Fold {i+1}: {auc:.4f}' for i, auc in enumerate(meta['xgboost']['fold_aucs']))}

### LightGBM Fold AUCs
{chr(10).join(f'- Fold {i+1}: {auc:.4f}' for i, auc in enumerate(meta['lightgbm']['fold_aucs']))}

## Risk Tiers

| Tier | Score Range | Action |
|---|---|---|
| {config['risk_tier_labels']['auto_approve']} | 0.00 – {config['risk_thresholds']['auto_approve']} | Release payouts |
| {config['risk_tier_labels']['enhanced_review']} | {config['risk_thresholds']['auto_approve']} – {config['risk_thresholds']['enhanced_review']} | Enhanced monitoring |
| {config['risk_tier_labels']['manual_review']} | {config['risk_thresholds']['enhanced_review']} – {config['risk_thresholds']['manual_review']} | Hold payouts, manual review |
| {config['risk_tier_labels']['auto_reject']} | {config['risk_thresholds']['manual_review']} – 1.00 | Freeze account immediately |

## Known Limitations

1. **Proxy Identities**: The UID (card1 + addr1 + D1) is a proxy for merchant identity. 
   In production, actual `merchant_id` would be used.
2. **Feature Anonymisation**: V-features are anonymised by Vesta — their business meaning is unknown.
3. **Dataset Vintage**: The IEEE-CIS dataset is from 2018-2019. Fraud patterns evolve.
4. **No Real-Time Features**: Features like "time since last transaction" require a feature store.

## Ethical Considerations

- Model should not be used as the sole decision-maker for merchant account actions.
- Human review is required for high-risk decisions (payout holds, account freezes).
- Regular monitoring for model drift and bias is essential.
- Merchants should have an appeals process for contested risk decisions.
"""
    
    card_path = os.path.join(model_dir, 'model_card.md')
    with open(card_path, 'w') as f:
        f.write(card)
    print(f"   📄 Model card saved: {card_path}")
