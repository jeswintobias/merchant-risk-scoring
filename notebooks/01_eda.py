# %% [markdown]
# # 📊 Exploratory Data Analysis — IEEE-CIS Transaction Data
# 
# ## Post-Onboarding Merchant Risk & Fraud Detection
# 
# **Purpose**: Standalone EDA notebook that explores the raw IEEE-CIS dataset 
# **before** any feature engineering or encoding. This ensures all visualisations 
# use original categorical labels and produce correct, publication-quality plots.
# 
# **Dataset**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) 
# — Real anonymised transaction data from Vesta Corporation.
# - 590,540 training transactions | 506,691 test transactions
# - 394 transaction features + 41 identity features
# - **3.50% fraud rate** (extreme class imbalance)
# 
# **Key Outputs**: All plots saved to `outputs/eda/` for integration into the 
# training pipeline notebook.

# %%
# ============================================================================
# IMPORTS & SETUP
# ============================================================================
import sys
import os
import gc
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if 'notebooks' in os.getcwd() else os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_ieee_data, print_system_info, get_output_dir

# ============================================================================
# COLOUR PALETTE — Razorpay-inspired dark theme
# ============================================================================
COLORS = {
    'primary': '#2563EB',      # Razorpay blue
    'secondary': '#7C3AED',    # Purple
    'success': '#059669',      # Green
    'danger': '#DC2626',       # Red
    'warning': '#D97706',      # Amber
    'bg_dark': '#0F172A',      # Dark background
    'bg_card': '#1E293B',      # Card background
    'text': '#F1F5F9',         # Light text
    'fraud': '#EF4444',        # Fraud colour
    'legit': '#22C55E',        # Legitimate colour
    'accent1': '#06B6D4',      # Cyan
    'accent2': '#F59E0B',      # Amber
    'accent3': '#EC4899',      # Pink
    'grid': '#334155',         # Subtle grid lines
}

# Consistent plot styling
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
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

EDA_DIR = get_output_dir('eda')
print(f"📁 EDA outputs will be saved to: {EDA_DIR}")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_system_info()

# %% [markdown]
# ---
# ## 1. Data Loading
# 
# We load only the training data for EDA (no test set needed — saves memory).
# The `load_ieee_data` function handles memory reduction via dtype downcasting.

# %%
# ============================================================================
# LOAD DATA (training only — test not needed for EDA)
# ============================================================================
data = load_ieee_data(load_test=False, verbose=True)
X_train = data['X_train']
y_train = data['y_train']

print(f"\n✅ Data loaded successfully")
print(f"   Shape: {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")

# %% [markdown]
# ---
# ## 2. Dataset Overview
# 
# Understanding the shape, types, and missing value patterns across 
# the 435 features before diving into specific analyses.

# %%
# ============================================================================
# DATASET OVERVIEW: DTYPES & MISSING VALUES
# ============================================================================
print("=" * 70)
print("  DATASET OVERVIEW")
print("=" * 70)

# Basic stats
n_rows, n_cols = X_train.shape
n_numeric = X_train.select_dtypes(include=[np.number]).shape[1]
n_categorical = X_train.select_dtypes(include=['object']).shape[1]

print(f"\n  Rows:          {n_rows:,}")
print(f"  Columns:       {n_cols}")
print(f"  Numeric:       {n_numeric}")
print(f"  Categorical:   {n_categorical}")
print(f"  Fraud rate:    {y_train.mean()*100:.2f}%")
print(f"  Fraud count:   {y_train.sum():,.0f} / {len(y_train):,}")

# Missing value analysis
missing = X_train.isnull().sum()
missing_pct = (missing / len(X_train) * 100).sort_values(ascending=False)
cols_with_missing = (missing_pct > 0).sum()
cols_over_50 = (missing_pct > 50).sum()
cols_over_90 = (missing_pct > 90).sum()

print(f"\n  Missing values:")
print(f"    Columns with any missing:  {cols_with_missing} / {n_cols}")
print(f"    Columns with >50% missing: {cols_over_50}")
print(f"    Columns with >90% missing: {cols_over_90}")

# %%
# ============================================================================
# MISSING VALUE HEATMAP — Top 50 columns by missingness
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))

# Get top 50 columns with most missing values
top_missing = missing_pct[missing_pct > 0].head(50)

bars = ax.barh(range(len(top_missing)), top_missing.values, color=COLORS['primary'], 
               edgecolor='none', alpha=0.85)

# Color bars by severity
for bar, pct in zip(bars, top_missing.values):
    if pct > 90:
        bar.set_color(COLORS['danger'])
    elif pct > 50:
        bar.set_color(COLORS['warning'])
    elif pct > 20:
        bar.set_color(COLORS['accent2'])

ax.set_yticks(range(len(top_missing)))
ax.set_yticklabels(top_missing.index, fontsize=8)
ax.set_xlabel('Missing %')
ax.set_title('Missing Value Distribution — Top 50 Columns\n'
             '🔴 >90%  🟡 >50%  🟠 >20%  🔵 <20%',
             fontweight='bold', pad=15)
ax.invert_yaxis()
ax.axvline(x=50, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=90, color=COLORS['danger'], linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'dataset_overview.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/dataset_overview.png")

# %% [markdown]
# ---
# ## 3. Target Analysis — Fraud Rate & Class Imbalance
# 
# The 3.50% fraud rate creates an extreme class imbalance (28:1 ratio).
# This mirrors real-world payment gateway data and requires careful handling 
# during model training (scale_pos_weight, stratified sampling).

# %%
# ============================================================================
# TARGET ANALYSIS: CLASS DISTRIBUTION & FRAUD RATE
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Class Distribution Bar ---
ax = axes[0]
counts = y_train.value_counts()
bars = ax.bar(['Legitimate', 'Fraudulent'], [counts[0], counts[1]], 
              color=[COLORS['legit'], COLORS['fraud']], edgecolor='white', linewidth=0.5,
              width=0.6)
for bar, count in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3000, 
            f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title('Class Distribution\n(Extreme 28:1 Imbalance)', fontweight='bold', pad=15)
ax.set_ylabel('Transaction Count')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

# --- Plot 2: Fraud Rate Pie ---
ax = axes[1]
fraud_rate = y_train.mean() * 100
wedges, texts, autotexts = ax.pie(
    [100 - fraud_rate, fraud_rate], 
    labels=['Legitimate', 'Fraud'],
    colors=[COLORS['legit'], COLORS['fraud']],
    autopct='%1.2f%%', startangle=90,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': COLORS['bg_dark'], 'linewidth': 2}
)
ax.set_title('Fraud Rate\n(3.50% — Mirrors Real Payment Gateways)', 
             fontweight='bold', pad=15)

# --- Plot 3: Transaction Amount Distribution by Class ---
ax = axes[2]
amt_legit = X_train.loc[y_train == 0, 'TransactionAmt']
amt_fraud = X_train.loc[y_train == 1, 'TransactionAmt']
ax.hist(amt_legit.clip(upper=1000), bins=100, alpha=0.7, color=COLORS['legit'], 
        label=f'Legitimate (n={len(amt_legit):,})', density=True)
ax.hist(amt_fraud.clip(upper=1000), bins=100, alpha=0.7, color=COLORS['fraud'], 
        label=f'Fraud (n={len(amt_fraud):,})', density=True)
ax.set_title('Transaction Amount Distribution\n(Clipped at $1,000)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Amount ($)')
ax.set_ylabel('Density')
ax.legend(labelcolor=COLORS['text'], fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/class_distribution.png")

# %% [markdown]
# ---
# ## 4. Transaction Amount Analysis
# 
# Deep dive into transaction amounts — the most intuitive risk signal. 
# Fraudulent transactions often have distinct amount patterns (round numbers, 
# very high values, specific cent patterns).

# %%
# ============================================================================
# TRANSACTION AMOUNT DEEP DIVE
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- Plot 1: Full Distribution (log scale) ---
ax = axes[0, 0]
ax.hist(np.log1p(amt_legit), bins=100, alpha=0.7, color=COLORS['legit'], 
        label='Legitimate', density=True)
ax.hist(np.log1p(amt_fraud), bins=100, alpha=0.7, color=COLORS['fraud'], 
        label='Fraud', density=True)
ax.set_title('Log-Transformed Amount Distribution', fontweight='bold', pad=10)
ax.set_xlabel('log(1 + Amount)')
ax.set_ylabel('Density')
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: Box Plot by Class ---
ax = axes[0, 1]
bp_data = [amt_legit.clip(upper=2000).dropna(), amt_fraud.clip(upper=2000).dropna()]
bp = ax.boxplot(bp_data, tick_labels=['Legitimate', 'Fraud'], patch_artist=True,
                medianprops={'color': COLORS['text'], 'linewidth': 2},
                whiskerprops={'color': COLORS['text']},
                capprops={'color': COLORS['text']},
                flierprops={'markerfacecolor': COLORS['text'], 'markersize': 2, 'alpha': 0.3})
bp['boxes'][0].set_facecolor(COLORS['legit'])
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(COLORS['fraud'])
bp['boxes'][1].set_alpha(0.7)
ax.set_title('Amount Box Plot by Class\n(Clipped at $2,000)', fontweight='bold', pad=10)
ax.set_ylabel('Amount ($)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 3: Percentile Comparison ---
ax = axes[1, 0]
percentiles = [10, 25, 50, 75, 90, 95, 99]
legit_pcts = [np.percentile(amt_legit.dropna(), p) for p in percentiles]
fraud_pcts = [np.percentile(amt_fraud.dropna(), p) for p in percentiles]

x = np.arange(len(percentiles))
width = 0.35
bars1 = ax.bar(x - width/2, legit_pcts, width, label='Legitimate', 
               color=COLORS['legit'], alpha=0.8)
bars2 = ax.bar(x + width/2, fraud_pcts, width, label='Fraud', 
               color=COLORS['fraud'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'P{p}' for p in percentiles])
ax.set_title('Amount Percentiles: Legitimate vs Fraud', fontweight='bold', pad=10)
ax.set_ylabel('Amount ($)')
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 4: Cents Analysis ---
ax = axes[1, 1]
cents_legit = (amt_legit - np.floor(amt_legit)).dropna()
cents_fraud = (amt_fraud - np.floor(amt_fraud)).dropna()

# Fraction of round-dollar transactions
round_legit = (cents_legit == 0).mean() * 100
round_fraud = (cents_fraud == 0).mean() * 100

bars = ax.bar(['Legitimate\n(Round $)', 'Fraud\n(Round $)', 
               'Legitimate\n(Has Cents)', 'Fraud\n(Has Cents)'],
              [round_legit, round_fraud, 100-round_legit, 100-round_fraud],
              color=[COLORS['legit'], COLORS['fraud'], 
                     COLORS['success'], COLORS['danger']],
              alpha=0.8, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, [round_legit, round_fraud, 100-round_legit, 100-round_fraud]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.set_title('Round-Dollar vs Cents Transactions\n(Fraud Indicator)', 
             fontweight='bold', pad=10)
ax.set_ylabel('Percentage (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'transaction_amounts.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/transaction_amounts.png")

# Print summary stats
print(f"\n📊 Amount Statistics:")
print(f"   Legitimate — Mean: ${amt_legit.mean():.2f} | Median: ${amt_legit.median():.2f} | Max: ${amt_legit.max():.2f}")
print(f"   Fraud      — Mean: ${amt_fraud.mean():.2f} | Median: ${amt_fraud.median():.2f} | Max: ${amt_fraud.max():.2f}")
print(f"   Round-dollar (legit): {round_legit:.1f}% | Round-dollar (fraud): {round_fraud:.1f}%")

# %% [markdown]
# ---
# ## 5. Product & Card Analysis
# 
# **Critical fix**: This section uses **raw categorical values** (W, H, C, S, R for ProductCD; 
# visa, mastercard, etc. for card4). The original notebook had this broken because 
# label encoding ran before EDA — we avoid that here by working on raw data.

# %%
# ============================================================================
# PRODUCT CODE & CARD TYPE ANALYSIS (RAW CATEGORICAL LABELS)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Fraud Rate by Product Code ---
ax = axes[0]
# Use raw categorical values — NOT encoded
product_fraud = pd.DataFrame({
    'fraud_rate': X_train.assign(isFraud=y_train).groupby('ProductCD')['isFraud'].mean() * 100,
    'count': X_train.groupby('ProductCD').size()
}).sort_values('fraud_rate', ascending=False)

bars = ax.bar(product_fraud.index.astype(str), product_fraud['fraud_rate'], 
              color=[COLORS['danger'] if r > 5 else COLORS['primary'] 
                     for r in product_fraud['fraud_rate']],
              edgecolor='white', linewidth=0.5, width=0.6)
for bar, (idx, row) in zip(bars, product_fraud.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{row["fraud_rate"]:.1f}%\n({row["count"]:,.0f})', 
            ha='center', fontweight='bold', fontsize=9)
ax.set_title('Fraud Rate by Product Code\n(W=Web, H=?, C=?, S=?, R=?)', 
             fontweight='bold', pad=15)
ax.set_ylabel('Fraud Rate (%)')
ax.axhline(y=y_train.mean()*100, color=COLORS['warning'], linestyle='--', alpha=0.7,
           label=f'Overall: {y_train.mean()*100:.2f}%')
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: Fraud Rate by Card Brand (card4) ---
ax = axes[1]
card4_fraud = pd.DataFrame({
    'fraud_rate': X_train.assign(isFraud=y_train).groupby('card4')['isFraud'].mean() * 100,
    'count': X_train.groupby('card4').size()
}).dropna().sort_values('fraud_rate', ascending=True)

bars = ax.barh(card4_fraud.index.astype(str), card4_fraud['fraud_rate'], 
               color=COLORS['secondary'], edgecolor='white', linewidth=0.5, height=0.6)
for bar, (idx, row) in zip(bars, card4_fraud.iterrows()):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2.,
            f'{row["fraud_rate"]:.1f}% (n={row["count"]:,.0f})', 
            va='center', fontweight='bold', fontsize=9)
ax.set_title('Fraud Rate by Card Brand\n(visa, mastercard, discover, amex)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Fraud Rate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 3: Fraud Rate by Card Category (card6) ---
ax = axes[2]
card6_fraud = pd.DataFrame({
    'fraud_rate': X_train.assign(isFraud=y_train).groupby('card6')['isFraud'].mean() * 100,
    'count': X_train.groupby('card6').size()
}).dropna().sort_values('fraud_rate', ascending=True)

bars = ax.barh(card6_fraud.index.astype(str), card6_fraud['fraud_rate'],
               color=COLORS['warning'], edgecolor='white', linewidth=0.5, height=0.6)
for bar, (idx, row) in zip(bars, card6_fraud.iterrows()):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2.,
            f'{row["fraud_rate"]:.1f}% (n={row["count"]:,.0f})', 
            va='center', fontweight='bold', fontsize=9)
ax.set_title('Fraud Rate by Card Category\n(credit, debit, charge, debit or credit)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Fraud Rate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'product_card_analysis.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/product_card_analysis.png")

# Print findings
print(f"\n📊 Product Code Analysis:")
for idx, row in product_fraud.iterrows():
    marker = "🔴" if row['fraud_rate'] > 5 else "🟢"
    print(f"   {marker} ProductCD={idx}: {row['fraud_rate']:.2f}% fraud rate ({row['count']:,.0f} transactions)")

# %% [markdown]
# ---
# ## 6. Temporal Patterns
# 
# **Critical fix**: The original notebook showed a flat line because 
# `TransactionDT` modulo computation had precision issues with float32. 
# We fix this by using float64 for the intermediate computation.
# 
# TransactionDT is a relative timestamp in seconds. We convert it to 
# hour-of-day, day-of-week, and approximate month.

# %%
# ============================================================================
# TEMPORAL FEATURES EXTRACTION (using float64 for precision)
# ============================================================================
# TransactionDT is seconds from a reference point
# Convert to meaningful time units using float64 to avoid precision loss
DT = X_train['TransactionDT'].astype(np.float64)

DT_hour = np.floor(DT / 3600) % 24       # Hour of day (0-23)
DT_day_of_week = np.floor(DT / (3600 * 24)) % 7  # Day of week (0-6)
DT_month = ((DT - 86400) / (30 * 24 * 3600)).astype(int)  # Approximate month

print(f"📊 Temporal features computed:")
print(f"   Hours range:  {DT_hour.min():.0f} - {DT_hour.max():.0f}")
print(f"   Unique hours: {DT_hour.nunique()}")
print(f"   Months range: {DT_month.min()} - {DT_month.max()}")

# %%
# ============================================================================
# TEMPORAL PATTERNS PLOTS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# --- Plot 1: Fraud Rate by Hour of Day ---
ax = axes[0, 0]
hourly_data = pd.DataFrame({'hour': DT_hour, 'isFraud': y_train.values})
hourly_fraud = hourly_data.groupby('hour')['isFraud'].agg(['mean', 'count'])
hourly_fraud['mean'] *= 100

ax.fill_between(hourly_fraud.index, hourly_fraud['mean'].values, alpha=0.3, color=COLORS['fraud'])
ax.plot(hourly_fraud.index, hourly_fraud['mean'].values, color=COLORS['fraud'], linewidth=2.5,
        marker='o', markersize=5, markerfacecolor=COLORS['fraud'], markeredgecolor='white',
        markeredgewidth=1)
ax.axhline(y=y_train.mean()*100, color=COLORS['warning'], linestyle='--', alpha=0.7,
           linewidth=1.5, label=f'Overall: {y_train.mean()*100:.2f}%')
ax.set_title('🕐 Fraud Rate by Hour of Day\n(Night-time = Higher Risk?)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Hour of Day (0 = midnight)')
ax.set_ylabel('Fraud Rate (%)')
ax.set_xticks(range(0, 24, 2))
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: Transaction Volume by Hour ---
ax = axes[0, 1]
ax.bar(hourly_fraud.index, hourly_fraud['count'].values, color=COLORS['primary'], 
       alpha=0.8, edgecolor='none')
ax.set_title('📊 Transaction Volume by Hour', fontweight='bold', pad=15)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Transactions')
ax.set_xticks(range(0, 24, 2))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 3: Fraud Rate by Day of Week ---
ax = axes[1, 0]
dow_data = pd.DataFrame({'dow': DT_day_of_week, 'isFraud': y_train.values})
dow_fraud = dow_data.groupby('dow')['isFraud'].mean() * 100

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# Only use labels for days that exist in the data
existing_days = sorted(dow_fraud.index.astype(int).tolist())
bars = ax.bar([day_names[d] if d < 7 else str(d) for d in existing_days], 
              [dow_fraud[d] for d in existing_days],
              color=COLORS['accent1'], edgecolor='white', linewidth=0.5, width=0.6)
for bar, d in zip(bars, existing_days):
    val = dow_fraud[d]
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=10)
ax.axhline(y=y_train.mean()*100, color=COLORS['warning'], linestyle='--', alpha=0.7,
           label=f'Overall: {y_train.mean()*100:.2f}%')
ax.set_title('📅 Fraud Rate by Day of Week', fontweight='bold', pad=15)
ax.set_ylabel('Fraud Rate (%)')
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 4: Fraud Rate by Month ---
ax = axes[1, 1]
monthly_data = pd.DataFrame({'month': DT_month, 'isFraud': y_train.values})
monthly_fraud = monthly_data.groupby('month').agg(
    fraud_rate=('isFraud', 'mean'),
    count=('isFraud', 'count')
)
monthly_fraud['fraud_rate'] *= 100

ax2 = ax.twinx()
ax.bar(monthly_fraud.index, monthly_fraud['count'].values, color=COLORS['primary'], 
       alpha=0.4, label='Transaction Volume')
ax2.plot(monthly_fraud.index, monthly_fraud['fraud_rate'].values, color=COLORS['fraud'], 
         linewidth=2.5, marker='s', markersize=6, label='Fraud Rate')
ax.set_title('📈 Monthly Transaction Volume & Fraud Rate', fontweight='bold', pad=15)
ax.set_xlabel('Month (approximate)')
ax.set_ylabel('Transaction Volume', color=COLORS['primary'])
ax2.set_ylabel('Fraud Rate (%)', color=COLORS['fraud'])
ax2.spines['right'].set_color(COLORS['fraud'])
ax2.tick_params(axis='y', colors=COLORS['fraud'])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax.spines['top'].set_visible(False)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, labelcolor=COLORS['text'], loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'temporal_patterns.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/temporal_patterns.png")

# %% [markdown]
# ---
# ## 7. Email Domain Analysis
# 
# **Critical fix**: Uses raw email domain strings (gmail.com, yahoo.com, etc.) 
# instead of encoded integers. Separated into its own plot (was crammed into 
# temporal patterns in the original).

# %%
# ============================================================================
# EMAIL DOMAIN ANALYSIS (RAW STRING VALUES)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Plot 1: P_emaildomain fraud rate (top 15 by volume) ---
ax = axes[0]
p_email_fraud = X_train.assign(isFraud=y_train).groupby('P_emaildomain').agg(
    fraud_rate=('isFraud', 'mean'),
    count=('isFraud', 'count')
)
# Filter to domains with significant volume
p_email_fraud = p_email_fraud[p_email_fraud['count'] > 500].sort_values('fraud_rate', ascending=True)
top_p_emails = p_email_fraud.tail(15)

bars = ax.barh(top_p_emails.index.astype(str), top_p_emails['fraud_rate'] * 100,
               color=COLORS['primary'], edgecolor='white', linewidth=0.5)

# Color high-risk domains
for bar, (idx, row) in zip(bars, top_p_emails.iterrows()):
    if row['fraud_rate'] * 100 > 10:
        bar.set_color(COLORS['danger'])
    elif row['fraud_rate'] * 100 > 5:
        bar.set_color(COLORS['warning'])
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2.,
            f'{row["fraud_rate"]*100:.1f}% (n={row["count"]:,.0f})', 
            va='center', fontweight='bold', fontsize=9)

ax.set_title('📧 Fraud Rate by Purchaser Email Domain\n(Top 15 by volume, min 500 txns)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Fraud Rate (%)')
ax.axvline(x=y_train.mean()*100, color=COLORS['warning'], linestyle='--', alpha=0.7,
           label=f'Overall: {y_train.mean()*100:.2f}%')
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: R_emaildomain analysis ---
ax = axes[1]
r_email_fraud = X_train.assign(isFraud=y_train).groupby('R_emaildomain').agg(
    fraud_rate=('isFraud', 'mean'),
    count=('isFraud', 'count')
)
r_email_fraud = r_email_fraud[r_email_fraud['count'] > 200].sort_values('fraud_rate', ascending=True)
top_r_emails = r_email_fraud.tail(15)

bars = ax.barh(top_r_emails.index.astype(str), top_r_emails['fraud_rate'] * 100,
               color=COLORS['accent1'], edgecolor='white', linewidth=0.5)
for bar, (idx, row) in zip(bars, top_r_emails.iterrows()):
    if row['fraud_rate'] * 100 > 10:
        bar.set_color(COLORS['danger'])
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2.,
            f'{row["fraud_rate"]*100:.1f}% (n={row["count"]:,.0f})',
            va='center', fontweight='bold', fontsize=9)

ax.set_title('📧 Fraud Rate by Recipient Email Domain\n(Top 15 by volume, min 200 txns)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Fraud Rate (%)')
ax.axvline(x=y_train.mean()*100, color=COLORS['warning'], linestyle='--', alpha=0.7,
           label=f'Overall: {y_train.mean()*100:.2f}%')
ax.legend(labelcolor=COLORS['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'email_domain_analysis.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/email_domain_analysis.png")

# Print key findings
print(f"\n📊 Email Domain Insights:")
print(f"   P_emaildomain missing: {X_train['P_emaildomain'].isnull().mean()*100:.1f}%")
print(f"   R_emaildomain missing: {X_train['R_emaildomain'].isnull().mean()*100:.1f}%")
if len(top_p_emails) > 0:
    highest = top_p_emails.iloc[-1]
    print(f"   Highest risk P_email: {top_p_emails.index[-1]} ({highest['fraud_rate']*100:.1f}%)")

# %% [markdown]
# ---
# ## 8. Identity & Device Analysis
# 
# Device type, browser, and screen resolution can reveal automated 
# or suspicious merchant activity patterns.

# %%
# ============================================================================
# IDENTITY & DEVICE ANALYSIS
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Device Type ---
ax = axes[0]
if 'DeviceType' in X_train.columns:
    device_fraud = X_train.assign(isFraud=y_train).groupby('DeviceType').agg(
        fraud_rate=('isFraud', 'mean'),
        count=('isFraud', 'count')
    ).dropna().sort_values('count', ascending=False)
    
    if len(device_fraud) > 0:
        bars = ax.bar(device_fraud.index.astype(str), device_fraud['fraud_rate'] * 100,
                      color=[COLORS['primary'], COLORS['secondary']][:len(device_fraud)],
                      edgecolor='white', linewidth=0.5, width=0.5)
        for bar, (idx, row) in zip(bars, device_fraud.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{row["fraud_rate"]*100:.2f}%\n(n={row["count"]:,.0f})',
                    ha='center', fontweight='bold', fontsize=10)
    ax.axhline(y=y_train.mean()*100, color=COLORS['warning'], linestyle='--', alpha=0.7)
ax.set_title('📱 Fraud Rate by Device Type', fontweight='bold', pad=15)
ax.set_ylabel('Fraud Rate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: Top Device Info by fraud rate ---
ax = axes[1]
if 'DeviceInfo' in X_train.columns:
    device_info_fraud = X_train.assign(isFraud=y_train).groupby('DeviceInfo').agg(
        fraud_rate=('isFraud', 'mean'),
        count=('isFraud', 'count')
    )
    # Filter to devices with enough data
    device_info_fraud = device_info_fraud[device_info_fraud['count'] > 200]
    top_devices = device_info_fraud.sort_values('fraud_rate', ascending=True).tail(10)
    
    if len(top_devices) > 0:
        bars = ax.barh(top_devices.index.astype(str), top_devices['fraud_rate'] * 100,
                       color=COLORS['accent3'], edgecolor='white', linewidth=0.5)
        for bar, (idx, row) in zip(bars, top_devices.iterrows()):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2.,
                    f'{row["fraud_rate"]*100:.1f}%', va='center', fontweight='bold', fontsize=9)
ax.set_title('📱 Top 10 Riskiest Devices\n(min 200 transactions)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Fraud Rate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 3: id_31 (Browser) Analysis ---
ax = axes[2]
if 'id_31' in X_train.columns:
    browser_fraud = X_train.assign(isFraud=y_train).groupby('id_31').agg(
        fraud_rate=('isFraud', 'mean'),
        count=('isFraud', 'count')
    )
    browser_fraud = browser_fraud[browser_fraud['count'] > 500]
    top_browsers = browser_fraud.sort_values('fraud_rate', ascending=True).tail(10)
    
    if len(top_browsers) > 0:
        bars = ax.barh(top_browsers.index.astype(str), top_browsers['fraud_rate'] * 100,
                       color=COLORS['accent1'], edgecolor='white', linewidth=0.5)
        for bar, (idx, row) in zip(bars, top_browsers.iterrows()):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2.,
                    f'{row["fraud_rate"]*100:.1f}%', va='center', fontweight='bold', fontsize=9)
ax.set_title('🌐 Top 10 Riskiest Browsers\n(min 500 transactions)', 
             fontweight='bold', pad=15)
ax.set_xlabel('Fraud Rate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'device_analysis.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/device_analysis.png")

# %% [markdown]
# ---
# ## 9. V-Feature Groups — NaN Patterns & Correlations
# 
# The V-features (V1-V339) are Vesta's proprietary engineered features. 
# They come in groups with similar NaN patterns, suggesting they were 
# computed from the same source data. Understanding these groups helps 
# with feature selection and dimensionality reduction.

# %%
# ============================================================================
# V-FEATURE NaN PATTERN ANALYSIS
# ============================================================================
v_cols = [c for c in X_train.columns if c.startswith('V')]
print(f"📊 Total V-features: {len(v_cols)}")

# Compute NaN percentage for each V-feature
v_nan_pct = X_train[v_cols].isnull().mean() * 100

# Group V-features by their NaN pattern
v_nan_groups = {}
for col in v_cols:
    pct = round(v_nan_pct[col], 1)
    if pct not in v_nan_groups:
        v_nan_groups[pct] = []
    v_nan_groups[pct].append(col)

print(f"\n📊 V-Feature NaN Groups:")
for pct in sorted(v_nan_groups.keys()):
    cols = v_nan_groups[pct]
    col_range = f"V{min(int(c[1:]) for c in cols)}-V{max(int(c[1:]) for c in cols)}"
    print(f"   {pct:5.1f}% missing: {len(cols):3d} features ({col_range})")

# %%
# ============================================================================
# V-FEATURE CORRELATION WITH FRAUD
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Plot 1: V-feature NaN pattern ---
ax = axes[0]
v_indices = [int(c[1:]) for c in v_cols]
ax.bar(v_indices, [v_nan_pct[f'V{i}'] for i in v_indices], 
       color=COLORS['primary'], edgecolor='none', alpha=0.7, width=1)
ax.set_title('V-Feature Missing Value Pattern\n(Groups share NaN rates → same source)', 
             fontweight='bold', pad=15)
ax.set_xlabel('V-Feature Index')
ax.set_ylabel('Missing %')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Plot 2: Top V-features correlated with fraud ---
ax = axes[1]
# Compute correlation of V-features with fraud label
v_fraud_corr = X_train[v_cols].corrwith(y_train).abs().sort_values(ascending=False)
top_v = v_fraud_corr.head(20)

bars = ax.barh(range(len(top_v)), top_v.values, color=COLORS['secondary'], 
               edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top_v)))
ax.set_yticklabels(top_v.index, fontsize=9)
for bar, val in zip(bars, top_v.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2.,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_title('Top 20 V-Features by |Correlation| with Fraud', 
             fontweight='bold', pad=15)
ax.set_xlabel('|Pearson Correlation|')
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'v_feature_correlations.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/v_feature_correlations.png")

# %% [markdown]
# ---
# ## 10. Feature vs Fraud Correlation — Top 20 Overall
# 
# Across all features (not just V-features), which have the strongest 
# linear relationship with the fraud label? This informs feature selection 
# and helps us understand what the model will likely rely on.

# %%
# ============================================================================
# TOP 20 FEATURES CORRELATED WITH FRAUD
# ============================================================================
# Compute correlation for all numeric features
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
all_corr = X_train[numeric_cols].corrwith(y_train).abs().sort_values(ascending=False)

# Remove TransactionDT (not a real feature)
all_corr = all_corr.drop('TransactionDT', errors='ignore')
top_20 = all_corr.head(20)

fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.barh(range(len(top_20)), top_20.values, edgecolor='white', linewidth=0.5)

# Color by feature group
for i, (feat, val) in enumerate(top_20.items()):
    if feat.startswith('V'):
        bars[i].set_color(COLORS['secondary'])
    elif feat.startswith('C'):
        bars[i].set_color(COLORS['accent1'])
    elif feat.startswith('D'):
        bars[i].set_color(COLORS['accent2'])
    elif feat.startswith('id_'):
        bars[i].set_color(COLORS['accent3'])
    else:
        bars[i].set_color(COLORS['primary'])
    
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20.index, fontsize=10)
ax.set_title('Top 20 Features by |Correlation| with Fraud\n'
             '🔵 Transaction  🟣 V-features  🔵 C-features  🟡 D-features  🩷 Identity',
             fontweight='bold', pad=15)
ax.set_xlabel('|Pearson Correlation|')
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'feature_fraud_correlation.png'), dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg_dark'])
plt.show()
print("✅ Saved: outputs/eda/feature_fraud_correlation.png")

# Print top features
print(f"\n📊 Top 10 Most Fraud-Correlated Features:")
for i, (feat, val) in enumerate(top_20.head(10).items(), 1):
    print(f"   {i:2d}. {feat:<20s} |r| = {val:.4f}")

# %% [markdown]
# ---
# ## 11. Key Insights Summary
# 
# These findings inform our feature engineering and model training decisions 
# in the main pipeline notebook (`02_merchant_risk_detection`).

# %%
# ============================================================================
# KEY INSIGHTS SUMMARY
# ============================================================================
print("=" * 70)
print("  📋 EDA KEY INSIGHTS — FOR TRAINING NOTEBOOK")
print("=" * 70)

print("""
1. CLASS IMBALANCE
   • 3.50% fraud rate (28:1 ratio) → Use scale_pos_weight in XGBoost/LightGBM
   • Consider PR-AUC as primary metric (more informative than ROC-AUC for rare events)

2. TRANSACTION AMOUNTS
   • Fraud transactions tend to have different amount distributions
   • Round-dollar transactions may correlate with fraud differently
   • Log-transform of TransactionAmt is useful for normalisation

3. PRODUCT & CARD ANALYSIS
   • ProductCD shows varying fraud rates across categories
   • Card brand and card type have distinct fraud signatures
   • These should be used as categorical features (not encoded before EDA!)

4. TEMPORAL PATTERNS
   • Fraud rate varies by hour of day — potential circadian risk signal
   • Monthly variation exists — may indicate seasonal fraud campaigns
   • Day-of-week effects may be present

5. EMAIL DOMAINS
   • Certain email domains have significantly higher fraud rates
   • Missing email domain itself is a signal
   • P_emaildomain and R_emaildomain carry different risk information

6. DEVICE & BROWSER
   • Device type (mobile vs desktop) correlates with fraud
   • Specific device models and browsers have elevated fraud rates
   • Browser fingerprint diversity could indicate automated attacks

7. V-FEATURES
   • 339 V-features group into NaN-pattern clusters (shared source)
   • Several V-features have strong linear correlation with fraud
   • High-dimensional — feature selection or tree-based models preferred

8. FEATURE ENGINEERING PRIORITIES
   • UID construction (card1 + addr1 + D1) for merchant identity
   • Aggregation features per UID (velocity, amount stats, device entropy)
   • Temporal features (hour, day-of-week)
   • Email domain risk signals
""")

print("=" * 70)
print("  ✅ EDA COMPLETE — All plots saved to outputs/eda/")
print("=" * 70)

# Clean up
del DT, DT_hour, DT_day_of_week, DT_month
gc.collect()
print(f"\n🧹 Memory cleaned up.")
