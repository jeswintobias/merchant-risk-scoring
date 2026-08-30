# 🛡️ Model Card — Post-Onboarding Merchant Risk Scorer

## Model Details

| Property | Value |
|---|---|
| **Model Type** | XGBoost + LightGBM Ensemble |
| **Problem** | Post-Onboarding Merchant Risk & Fraud Detection |
| **Dataset** | IEEE-CIS Fraud Detection (Vesta Corporation) |
| **Training Rows** | 590,540 |
| **Features** | 434 |
| **Trained At** | 2026-08-30T15:56:09.867136 |
| **System** | arm64 — Darwin |

## Performance

| Metric | Value |
|---|---|
| **Ensemble AUC-ROC** | 0.9520 |
| **XGBoost Mean AUC** | 0.9477 ± 0.0108 |
| **LightGBM Mean AUC** | 0.9472 ± 0.0115 |
| **XGBoost Weight** | 0.95 |
| **LightGBM Weight** | 0.05 |

## Cross-Validation

- **Strategy**: GroupKFold by month
- **Folds**: 6
- **Group Key**: DT_M (approximate month)

### XGBoost Fold AUCs
- Fold 1: 0.9246
- Fold 2: 0.9518
- Fold 3: 0.9543
- Fold 4: 0.9569
- Fold 5: 0.9469
- Fold 6: 0.9518

### LightGBM Fold AUCs
- Fold 1: 0.9242
- Fold 2: 0.9539
- Fold 3: 0.9533
- Fold 4: 0.9567
- Fold 5: 0.9409
- Fold 6: 0.9545

## Risk Tiers

| Tier | Score Range | Action |
|---|---|---|
| 🟢 LOW RISK — Release Payouts | 0.00 – 0.05 | Release payouts |
| 🟡 MEDIUM RISK — Enhanced Monitoring | 0.05 – 0.2 | Enhanced monitoring |
| 🟠 HIGH RISK — Hold Payouts for Review | 0.2 – 0.5 | Hold payouts, manual review |
| 🔴 CRITICAL RISK — Freeze Account | 0.5 – 1.00 | Freeze account immediately |

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
