"""
data_loader.py — Memory-optimised data loading for IEEE-CIS Fraud Detection dataset.

Shared module used by both the EDA notebook and the training pipeline.
Handles dtype reduction, path resolution, and system info detection.
"""

import os
import sys
import time
import platform
import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATH RESOLUTION
# ============================================================================
def get_project_root():
    """Auto-detect the project root directory regardless of where the script is called from."""
    # Try to resolve from this file's location (src/ is one level down from root)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    
    # Verify by checking for the data directory
    if os.path.exists(os.path.join(project_root, 'data')):
        return project_root
    
    # Fallback: try cwd
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        return os.getcwd()
    
    raise FileNotFoundError(
        "Cannot locate project root. Expected 'data/' directory. "
        "Run from the project root or ensure folder structure is correct."
    )


def get_data_dir():
    """Return the path to the IEEE-CIS dataset directory."""
    return os.path.join(get_project_root(), 'data', 'ieee-fraud-detection')


def get_output_dir(subdir='eda'):
    """Return the path to an output subdirectory, creating it if needed."""
    path = os.path.join(get_project_root(), 'outputs', subdir)
    os.makedirs(path, exist_ok=True)
    return path


def get_model_dir(subdir='final'):
    """Return the path to a model subdirectory, creating it if needed."""
    path = os.path.join(get_project_root(), 'models', subdir)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# SYSTEM INFORMATION
# ============================================================================
def get_system_info():
    """Auto-detect system info instead of hardcoding hardware."""
    info = {
        'machine': platform.machine(),
        'system': platform.system(),
        'release': platform.release(),
        'processor': platform.processor() or 'Unknown',
        'python_version': sys.version.split()[0],
    }
    
    # Try to get RAM info
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info['ram_gb'] = 'Unknown (install psutil for detection)'
    
    return info


def print_system_info():
    """Print formatted system information."""
    info = get_system_info()
    print(f"🖥️  Running on: {info['machine']} — {info['system']} {info['release']}")
    print(f"💾 RAM: {info['ram_gb']} GB")
    print(f"🐍 Python: {info['python_version']}")
    try:
        print(f"📊 Pandas: {pd.__version__} | NumPy: {np.__version__}")
    except Exception:
        pass


# ============================================================================
# MEMORY REDUCTION
# ============================================================================
def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all numeric columns and downcast dtypes to reduce memory usage.
    Critical for running on machines with limited RAM.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimise.
    verbose : bool
        If True, print memory reduction stats.
    
    Returns
    -------
    pd.DataFrame
        Memory-optimised DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f'  Memory: {start_mem:.1f} MB → {end_mem:.1f} MB ({reduction:.1f}% reduction)')
    return df


# ============================================================================
# DATA LOADING
# ============================================================================
def load_ieee_data(load_test=True, verbose=True):
    """
    Load the IEEE-CIS Fraud Detection dataset with memory optimisation.
    
    Parameters
    ----------
    load_test : bool
        If True, also load test data. Set to False for EDA-only (saves memory).
    verbose : bool
        If True, print loading progress.
    
    Returns
    -------
    dict with keys:
        'X_train' : pd.DataFrame — Training features (merged transaction + identity)
        'y_train' : pd.Series — Target variable (isFraud)
        'X_test'  : pd.DataFrame or None — Test features (if load_test=True)
    """
    data_dir = get_data_dir()
    
    if verbose:
        print("=" * 70)
        print("  LOADING IEEE-CIS FRAUD DETECTION DATASET")
        print("=" * 70)
    
    t0 = time.time()
    
    # --- Train Transaction ---
    if verbose:
        print("\n📂 Loading train_transaction.csv...")
    train_txn = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
    train_txn = reduce_mem_usage(train_txn, verbose=verbose)
    
    # --- Train Identity ---
    if verbose:
        print("\n📂 Loading train_identity.csv...")
    train_id = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
    train_id = reduce_mem_usage(train_id, verbose=verbose)
    
    # --- Test Data (optional) ---
    test_txn = None
    test_id = None
    if load_test:
        if verbose:
            print("\n📂 Loading test_transaction.csv...")
        test_txn = pd.read_csv(os.path.join(data_dir, 'test_transaction.csv'))
        test_txn = reduce_mem_usage(test_txn, verbose=verbose)
        
        if verbose:
            print("\n📂 Loading test_identity.csv...")
        test_id = pd.read_csv(os.path.join(data_dir, 'test_identity.csv'))
        test_id = reduce_mem_usage(test_id, verbose=verbose)
    
    if verbose:
        print(f"\n⏱️  Total load time: {time.time()-t0:.1f}s")
    
    # --- Merge Transaction + Identity ---
    if verbose:
        print("\n🔗 Merging Transaction + Identity tables...")
    
    X_train = train_txn.merge(train_id, on='TransactionID', how='left')
    del train_txn, train_id
    
    X_train.set_index('TransactionID', drop=True, inplace=True)
    y_train = X_train['isFraud'].copy()
    X_train.drop('isFraud', axis=1, inplace=True)
    
    X_test = None
    if load_test and test_txn is not None:
        X_test = test_txn.merge(test_id, on='TransactionID', how='left')
        del test_txn, test_id
        X_test.set_index('TransactionID', drop=True, inplace=True)
    
    import gc
    gc.collect()
    
    if verbose:
        print(f"\n📊 Final Dataset Shapes:")
        print(f"   X_train: {X_train.shape[0]:,} × {X_train.shape[1]}")
        if X_test is not None:
            print(f"   X_test:  {X_test.shape[0]:,} × {X_test.shape[1]}")
        print(f"   y_train: {y_train.shape[0]:,} (fraud rate: {y_train.mean()*100:.2f}%)")
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
    }
    
    return result
