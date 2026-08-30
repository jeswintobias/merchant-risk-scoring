"""
feature_engineering.py — Feature engineering pipeline for merchant risk scoring.

Contains all feature construction functions used by the training pipeline:
- Categorical encoding (label encoding)
- Magic UID construction (card1 + addr1 + D1)
- Frequency encoding
- Group aggregation (mean, std, nunique)
- Temporal features
- Outsider flags

All functions are designed to save their artifacts (encoders, maps, dicts) to
models/artifacts/ so they can be reloaded for inference on new data.
"""

import os
import time
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================
def build_temporal_features(X_train, X_test=None):
    """
    Extract temporal features from TransactionDT.
    Uses float64 for intermediate computation to avoid precision loss.
    
    Creates: DT_hour, DT_day, DT_M (month)
    """
    print("\n🕐 Building temporal features...")
    
    for df, name in [(X_train, 'train')] + ([(X_test, 'test')] if X_test is not None else []):
        DT = df['TransactionDT'].astype(np.float64)
        df['DT_hour'] = np.floor(DT / 3600) % 24
        df['DT_day']  = np.floor(DT / (3600 * 24))
        df['DT_M']    = ((DT - 86400) / (30 * 24 * 3600)).astype(int)
        print(f"   ✓ Temporal features for {name}: hours 0-23, {df['DT_M'].nunique()} months")
    
    return X_train, X_test


# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================
def encode_categoricals(X_train, X_test=None, save_dir=None):
    """
    Label encode categorical columns. Fits on combined train+test to handle 
    unseen categories. Saves encoders if save_dir is provided.
    
    Returns the label encoders dict for later use in inference.
    """
    print("\n🏷️  Label encoding categorical features...")
    
    # First create combined card features (before encoding)
    print("   🔧 Creating combined card features...")
    X_train['card1_addr1'] = X_train['card1'].astype(str) + '_' + X_train['addr1'].astype(str)
    X_train['card1_addr1_P_email'] = X_train['card1_addr1'] + '_' + X_train['P_emaildomain'].astype(str)
    
    if X_test is not None:
        X_test['card1_addr1'] = X_test['card1'].astype(str) + '_' + X_test['addr1'].astype(str)
        X_test['card1_addr1_P_email'] = X_test['card1_addr1'] + '_' + X_test['P_emaildomain'].astype(str)
    
    # Extract cents and log-transform
    X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
    X_train['TransactionAmt_log'] = np.log1p(X_train['TransactionAmt'])
    
    if X_test is not None:
        X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
        X_test['TransactionAmt_log'] = np.log1p(X_test['TransactionAmt'])
    
    print("   ✓ Combined card features, cents, log-transform created")
    
    # Label encode categorical columns
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
                'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
                'DeviceType', 'DeviceInfo',
                'card1_addr1', 'card1_addr1_P_email']
    
    label_encoders = {}
    encoded_count = 0
    
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            col_in_test = X_test is not None and col in X_test.columns
            if col_in_test:
                combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
            else:
                combined = X_train[col].astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            if col_in_test:
                X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
            encoded_count += 1
    
    print(f"   ✓ {encoded_count} categorical columns label-encoded")
    
    # Save encoders
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"   💾 Label encoders saved to {save_dir}/label_encoders.pkl")
    
    return X_train, X_test, label_encoders


# ============================================================================
# MAGIC UID FEATURE
# ============================================================================
def build_uid_features(X_train, X_test=None):
    """
    Construct the 'Magic UID' — a pseudo merchant/client identity from card1 + addr1 + D1.
    
    This is the core innovation from the 1st place Kaggle solution. In a real Razorpay 
    context, this would be replaced by actual merchant_id. The UID allows us to compute 
    merchant-level aggregation features from transaction-level data.
    """
    print("\n🪄 Constructing Magic UID Feature (Pseudo Merchant Identity)...")
    
    X_train['day'] = X_train['TransactionDT'] / (24 * 60 * 60)
    X_train['uid'] = X_train['card1_addr1'].astype(str) + '_' + np.floor(X_train['day'] - X_train['D1']).astype(str)
    
    if X_test is not None:
        X_test['day'] = X_test['TransactionDT'] / (24 * 60 * 60)
        X_test['uid'] = X_test['card1_addr1'].astype(str) + '_' + np.floor(X_test['day'] - X_test['D1']).astype(str)
    
    n_train = X_train['uid'].nunique()
    print(f"   ✓ UID created for train: {n_train:,} unique identities")
    if X_test is not None:
        n_test = X_test['uid'].nunique()
        print(f"   ✓ UID created for test:  {n_test:,} unique identities")
    
    return X_train, X_test


# ============================================================================
# FREQUENCY ENCODING
# ============================================================================
def encode_frequency(X_train, X_test, cols, save_dir=None):
    """
    Replace categories with their frequency (occurrence proportion) in training data.
    Saves frequency maps for inference.
    """
    freq_maps = {}
    
    for col in cols:
        vc = X_train[col].value_counts(dropna=True, normalize=True).to_dict()
        nm = col + '_FE'
        X_train[nm] = X_train[col].map(vc).astype('float32')
        X_test[nm]  = X_test[col].map(vc).astype('float32')
        X_test[nm].fillna(0, inplace=True)
        freq_maps[col] = vc
        print(f'   ✓ Frequency encoded: {col}')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'frequency_maps.pkl'), 'wb') as f:
            pickle.dump(freq_maps, f)
        print(f"   💾 Frequency maps saved to {save_dir}/frequency_maps.pkl")
    
    return X_train, X_test, freq_maps


# ============================================================================
# GROUP AGGREGATION
# ============================================================================
def encode_aggregation(main_cols, uids, aggregations, train_df, test_df, 
                       fillna=True, usena=False):
    """
    Calculate statistics (mean, std) per UID group.
    This captures merchant-level transaction velocity & behaviour patterns.
    """
    for main_col in main_cols:
        for uid_col in uids:
            dfs_to_concat = []
            if uid_col in train_df.columns and main_col in train_df.columns:
                dfs_to_concat.append(train_df[[uid_col, main_col]])
            if test_df is not None and uid_col in test_df.columns and main_col in test_df.columns:
                dfs_to_concat.append(test_df[[uid_col, main_col]])
            if not dfs_to_concat:
                continue
            
            for agg_type in aggregations:
                new_col = f'{main_col}_{uid_col}_{agg_type}'
                temp = pd.concat(dfs_to_concat, axis=0)
                if usena:
                    temp.loc[temp[main_col] == -1, main_col] = np.nan
                temp = temp.groupby(uid_col)[main_col].agg([agg_type]).reset_index()
                temp.columns = [uid_col, new_col]
                temp.index = list(temp[uid_col])
                temp = temp[new_col].to_dict()
                
                train_df[new_col] = train_df[uid_col].map(temp).astype('float32')
                test_df[new_col]  = test_df[uid_col].map(temp).astype('float32')
                
                if fillna:
                    train_df[new_col].fillna(-1, inplace=True)
                    test_df[new_col].fillna(-1, inplace=True)
    
    print(f'   ✓ Aggregated {main_cols} by {uids} with {aggregations}')
    return train_df, test_df


def encode_nunique(main_cols, uids, train_df, test_df):
    """Count distinct values per UID group (e.g., how many email domains per merchant)."""
    for main_col in main_cols:
        for uid_col in uids:
            dfs_to_concat = []
            if uid_col in train_df.columns and main_col in train_df.columns:
                dfs_to_concat.append(train_df[[uid_col, main_col]])
            if test_df is not None and uid_col in test_df.columns and main_col in test_df.columns:
                dfs_to_concat.append(test_df[[uid_col, main_col]])
            
            if not dfs_to_concat:
                continue
                
            comb = pd.concat(dfs_to_concat, axis=0)
            mp = comb.groupby(uid_col)[main_col].agg(['nunique'])
            mp.columns = [f'{main_col}_{uid_col}_ct']
            col_name = f'{main_col}_{uid_col}_ct'
            if uid_col in train_df.columns:
                train_df[col_name] = train_df[uid_col].map(mp[col_name]).astype('float32')
            if test_df is not None and uid_col in test_df.columns:
                test_df[col_name] = test_df[uid_col].map(mp[col_name]).astype('float32')
    
    print(f'   ✓ Nunique aggregated {main_cols} by {uids}')
    return train_df, test_df


# ============================================================================
# FULL FEATURE ENGINEERING PIPELINE
# ============================================================================
def run_feature_engineering(X_train, X_test, y_train, save_dir=None):
    """
    Run the complete feature engineering pipeline:
    1. Temporal features
    2. Categorical encoding (+ combined card features, cents, log-amt)
    3. UID construction
    4. Frequency encoding
    5. Group aggregation (47+ features)
    6. Outsider flags
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train : pd.Series (not used for engineering, but needed for reference)
    save_dir : str or None — path to save artifacts (e.g., 'models/artifacts')
    
    Returns
    -------
    X_train, X_test : pd.DataFrame (with new features)
    artifacts : dict — all saved objects (encoders, maps, etc.)
    """
    print("=" * 70)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    t0 = time.time()
    artifacts = {}
    
    # 1. Temporal features
    X_train, X_test = build_temporal_features(X_train, X_test)
    
    # 2. Categorical encoding (also creates card1_addr1, cents, log-amt)
    X_train, X_test, label_encoders = encode_categoricals(X_train, X_test, save_dir=save_dir)
    artifacts['label_encoders'] = label_encoders
    
    # 3. UID construction
    X_train, X_test = build_uid_features(X_train, X_test)
    
    # 4. Frequency encode UID
    print("\n📊 Building Group Aggregation Features (47+ new features)...")
    print("   This captures merchant-level transaction velocity & behaviour patterns\n")
    
    X_train, X_test, freq_maps = encode_frequency(X_train, X_test, ['uid'], save_dir=save_dir)
    artifacts['freq_maps'] = freq_maps
    
    # 5. Group aggregations
    # Amount and timedelta features by UID
    X_train, X_test = encode_aggregation(
        ['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['uid'], 
        ['mean', 'std'], X_train, X_test, fillna=True, usena=True)
    
    # Counting features by UID
    X_train, X_test = encode_aggregation(
        ['C' + str(x) for x in range(1, 15) if x != 3], ['uid'], 
        ['mean'], X_train, X_test, fillna=True, usena=True)
    
    # Match/mismatch features by UID
    X_train, X_test = encode_aggregation(
        ['M' + str(x) for x in range(1, 10)], ['uid'], 
        ['mean'], X_train, X_test, fillna=True, usena=True)
    
    # Nunique aggregations
    X_train, X_test = encode_nunique(
        ['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents'], ['uid'], X_train, X_test)
    
    # Additional aggregations
    X_train, X_test = encode_aggregation(
        ['C14'], ['uid'], ['std'], X_train, X_test, fillna=True, usena=True)
    X_train, X_test = encode_nunique(['C13', 'V314'], ['uid'], X_train, X_test)
    X_train, X_test = encode_nunique(
        ['V127', 'V136', 'V309', 'V307', 'V320'], ['uid'], X_train, X_test)
    
    # 6. Outsider flag
    X_train['outsider15'] = (np.abs(X_train['D1'] - X_train['D15']) > 3).astype('int8')
    X_test['outsider15']  = (np.abs(X_test['D1'] - X_test['D15']) > 3).astype('int8')
    print('   ✓ Outsider15 flag (D1-D15 inconsistency)')
    
    elapsed = time.time() - t0
    print(f"\n⏱️  Feature engineering time: {elapsed:.1f}s")
    print(f"📊 Features after engineering: {X_train.shape[1]}")
    
    return X_train, X_test, artifacts


# ============================================================================
# FEATURE SELECTION
# ============================================================================
def select_features(X_train):
    """
    Select features for training by removing:
    - Non-feature columns (TransactionDT, etc.)
    - Helper columns (uid, day, DT_hour, etc.)
    - Features that failed time consistency tests
    
    Returns
    -------
    cols : list — selected feature column names
    """
    print("\n🎯 Feature Selection & Time Consistency Filtering...")
    
    cols = list(X_train.columns)
    initial_count = len(cols)
    
    # Remove non-feature columns
    for c in ['TransactionDT', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
        if c in cols:
            cols.remove(c)
    
    # Remove helper columns
    for c in ['DT_M', 'day', 'uid', 'DT_hour', 'DT_day', 'oof_xgb', 'oof_lgb']:
        if c in cols:
            cols.remove(c)
    
    # Remove features that FAILED time consistency test
    failed = ['C3', 'M5', 'id_08', 'id_33', 'card4',
              'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']
    for c in failed:
        if c in cols:
            cols.remove(c)
    
    # Remove highly sparse identity columns
    for c in ['id_' + str(x) for x in range(22, 28)]:
        if c in cols:
            cols.remove(c)
    
    print(f"   ✅ Final feature count: {len(cols)} features selected")
    print(f"   ❌ Removed {initial_count - len(cols)} features (time-inconsistent / helper)")
    
    return cols


# ============================================================================
# ARTIFACT MANAGEMENT
# ============================================================================
def save_feature_list(cols, save_dir):
    """Save the final feature list as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'model_features.json'), 'w') as f:
        json.dump(cols, f)
    print(f"   💾 Feature list ({len(cols)} features) saved to {save_dir}/model_features.json")


def load_artifacts(artifact_dir):
    """Load all saved feature engineering artifacts for inference."""
    artifacts = {}
    
    le_path = os.path.join(artifact_dir, 'label_encoders.pkl')
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            artifacts['label_encoders'] = pickle.load(f)
    
    fm_path = os.path.join(artifact_dir, 'frequency_maps.pkl')
    if os.path.exists(fm_path):
        with open(fm_path, 'rb') as f:
            artifacts['freq_maps'] = pickle.load(f)
    
    feat_path = os.path.join(artifact_dir, 'model_features.json')
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            artifacts['feature_list'] = json.load(f)
    
    print(f"   📦 Loaded artifacts: {list(artifacts.keys())}")
    return artifacts
