"""
inference.py — MerchantRiskScorer for production inference.

Loads the trained ensemble model and generates risk scores with explanations.
Used by the FastAPI backend for real-time scoring.
"""

import os
import json
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import shap


class MerchantRiskScorer:
    """
    Production inference wrapper for the XGBoost + LightGBM ensemble.
    
    Usage:
        scorer = MerchantRiskScorer('models/final')
        result = scorer.score(features_df)
    """
    
    def __init__(self, model_dir):
        """
        Load models and configuration from model_dir.
        
        Parameters
        ----------
        model_dir : str
            Path to the final model directory (e.g., 'models/final')
        """
        self.model_dir = model_dir
        
        # Load XGBoost
        xgb_path = os.path.join(model_dir, 'xgb_final.json')
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(xgb_path)
        
        # Load LightGBM
        lgb_path = os.path.join(model_dir, 'lgb_final.txt')
        self.lgb_model = lgb.Booster(model_file=lgb_path)
        
        # Load SHAP model
        shap_path = os.path.join(model_dir, 'xgb_shap.json')
        if os.path.exists(shap_path):
            self.shap_model = xgb.XGBClassifier()
            self.shap_model.load_model(shap_path)
            self.explainer = shap.TreeExplainer(self.shap_model)
        else:
            self.shap_model = None
            self.explainer = None
        
        # Load ensemble config
        config_path = os.path.join(model_dir, 'ensemble_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.xgb_weight = self.config['xgb_weight']
        self.lgb_weight = self.config['lgb_weight']
        self.thresholds = self.config['risk_thresholds']
        self.tier_labels = self.config['risk_tier_labels']
        
        # Load feature list
        artifact_dir = os.path.join(os.path.dirname(model_dir), 'artifacts')
        feat_path = os.path.join(artifact_dir, 'model_features.json')
        if os.path.exists(feat_path):
            with open(feat_path, 'r') as f:
                self.feature_list = json.load(f)
        else:
            self.feature_list = None
        
        print(f"✅ MerchantRiskScorer loaded from {model_dir}")
        print(f"   XGBoost weight: {self.xgb_weight:.2f}")
        print(f"   LightGBM weight: {self.lgb_weight:.2f}")
        if self.feature_list:
            print(f"   Features: {len(self.feature_list)}")
    
    def score(self, X, explain=False):
        """
        Score a batch of transactions/merchants.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features DataFrame (must contain columns matching training features)
        explain : bool
            If True, compute SHAP explanations (slower)
        
        Returns
        -------
        dict with keys:
            'risk_scores': np.array of risk scores (0-1)
            'risk_tiers': list of tier labels
            'risk_actions': list of action strings
            'xgb_scores': np.array of XGBoost-only scores
            'lgb_scores': np.array of LightGBM-only scores
            'explanations': list of dicts (if explain=True)
        """
        # Work on a copy to avoid mutating the caller's DataFrame
        X = X.copy()
        
        cols = self.feature_list if self.feature_list else list(X.columns)
        
        # Ensure all features exist, fill missing with -1
        for col in cols:
            if col not in X.columns:
                X[col] = -1
        
        # Score with both models
        xgb_scores = self.xgb_model.predict_proba(X[cols])[:, 1]
        lgb_scores = self.lgb_model.predict(X[cols])
        
        # Ensemble blend
        risk_scores = self.xgb_weight * xgb_scores + self.lgb_weight * lgb_scores
        
        # Clamp to [0, 1] range for safety
        risk_scores = np.clip(risk_scores, 0.0, 1.0)
        
        # Classify into risk tiers
        risk_tiers = []
        risk_actions = []
        for score in risk_scores:
            tier, action = self._classify_risk(score)
            risk_tiers.append(tier)
            risk_actions.append(action)
        
        result = {
            'risk_scores': risk_scores,
            'risk_tiers': risk_tiers,
            'risk_actions': risk_actions,
            'xgb_scores': xgb_scores,
            'lgb_scores': lgb_scores,
        }
        
        # SHAP explanations
        if explain and self.explainer is not None:
            try:
                explanations = self._explain(X[cols])
                result['explanations'] = explanations
            except Exception as e:
                print(f"⚠️  SHAP explanation failed: {e}")
                result['explanations'] = []
        
        return result
    
    def _classify_risk(self, score):
        """Classify a risk score into an operational tier."""
        if score < self.thresholds['auto_approve']:
            return 'LOW', self.tier_labels['auto_approve']
        elif score < self.thresholds['enhanced_review']:
            return 'MEDIUM', self.tier_labels['enhanced_review']
        elif score < self.thresholds['manual_review']:
            return 'HIGH', self.tier_labels['manual_review']
        else:
            return 'CRITICAL', self.tier_labels['auto_reject']
    
    def _explain(self, X):
        """Generate SHAP explanations for each prediction."""
        shap_values = self.explainer.shap_values(X)
        explanations = []
        
        for i in range(len(X)):
            # Get top contributing features
            feature_contributions = list(zip(
                X.columns.tolist(),
                shap_values[i].tolist(),
                X.iloc[i].values.tolist()
            ))
            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_factors = []
            for feat_name, shap_val, feat_val in feature_contributions[:10]:
                top_factors.append({
                    'feature': feat_name,
                    'shap_value': float(shap_val),
                    'feature_value': float(feat_val) if not np.isnan(feat_val) else None,
                    'direction': 'increases_risk' if shap_val > 0 else 'decreases_risk'
                })
            
            explanations.append({
                'base_value': float(self.explainer.expected_value),
                'top_factors': top_factors
            })
        
        return explanations
    
    def get_model_info(self):
        """Return model metadata for API responses."""
        return {
            'model_type': 'XGBoost + LightGBM Ensemble',
            'xgb_weight': self.xgb_weight,
            'lgb_weight': self.lgb_weight,
            'ensemble_auc': self.config.get('ensemble_auc'),
            'feature_count': len(self.feature_list) if self.feature_list else None,
            'risk_thresholds': self.thresholds,
            'risk_tiers': self.tier_labels,
        }
