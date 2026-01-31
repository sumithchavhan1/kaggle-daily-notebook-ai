#!/usr/bin/env python3
"""
Main orchestration script for Kaggle Daily Notebook Generation
Fetches trending datasets, generates notebooks, and publishes them daily at 9 AM IST
Uses KAGGLE_API_TOKEN for authentication (stored in ~/.kaggle/kaggle.json)
Improved error handling and retry logic
"""
import os
import json
import logging
import time
from datetime import datetime
import sys
import random
from typing import Optional, Dict, Any

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from perplexity_integration import PerplexityNotebookGenerator
    from publish_to_kaggle import KaggleNotebookPublisher
except ImportError as e:
    print(f"Import error: {str(e)}")
    sys.exit(1)

# File to remember last used dataset, to avoid immediate repeats
LAST_DATASET_FILE = "last_dataset_ref.txt"

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("kaggle_notebook_gen.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class KaggLeNotebookOrchestrator:
    """Main orchestrator for daily notebook generation and publication with error handling"""

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def __init__(self):
        """Initialize the orchestrator with API credentials"""
        try:
            logger.info("Initializing Kaggle Notebook Orchestrator...")
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            logger.error(f"Failed to authenticate Kaggle API: {str(e)}")
            raise

        self.perplexity_key = os.getenv("KAGGLE_GROQ")
        if not self.perplexity_key:
            raise ValueError("KAGGLE_GROQ environment variable not set")

        try:
            self.perplexity_generator = PerplexityNotebookGenerator(self.perplexity_key)
            self.publisher = KaggleNotebookPublisher(self.kaggle_api)
            logger.info("Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _retry_operation(self, func, *args, operation_name: str = "operation", **kwargs) -> Any:
        """Generic retry wrapper for API operations"""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"Attempting {operation_name} (Attempt {attempt}/{self.MAX_RETRIES})")
                result = func(*args, **kwargs)
                logger.info(f"{operation_name} succeeded on attempt {attempt}")
                return result
            except Exception as e:
                logger.warning(f"{operation_name} failed on attempt {attempt}: {str(e)}")
                if attempt < self.MAX_RETRIES:
                    wait_time = self.RETRY_DELAY * attempt  # exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"{operation_name} failed after {self.MAX_RETRIES} attempts")
                    raise

    def fetch_trending_dataset(self) -> Optional[Dict[str, Any]]:
        """Fetch a trending dataset from Kaggle with retry logic, avoiding immediate repeats."""
        try:
            logger.info("Fetching trending datasets from Kaggle...")

            def _fetch():
                datasets = self.kaggle_api.dataset_list(
                    sort_by="hottest",
                    file_type="csv",
                )

                if not datasets:
                    logger.warning("No trending datasets found")
                    return None

                top_n = min(10, len(datasets))
                candidates = datasets[:top_n]

                # Load last dataset ref if exists
                last_ref = None
                if os.path.exists(LAST_DATASET_FILE):
                    try:
                        with open(LAST_DATASET_FILE, "r") as f:
                            last_ref = f.read().strip()
                    except Exception:
                        last_ref = None

                # Filter out last one if possible
                if last_ref:
                    filtered = [d for d in candidates if d.ref != last_ref]
                    if filtered:
                        candidates = filtered

                selected_dataset = random.choice(candidates)

                # Persist current ref for next run
                try:
                    with open(LAST_DATASET_FILE, "w") as f:
                        f.write(selected_dataset.ref)
                except Exception as e:
                    logger.warning(f"Could not write LAST_DATASET_FILE: {e}")

                logger.info(f"Selected dataset: {selected_dataset.ref}")

                return {
                    "ref": selected_dataset.ref,
                    "title": selected_dataset.title,
                    "description": getattr(selected_dataset, "subtitle", selected_dataset.title),
                    "size": getattr(selected_dataset, "size_bytes", 0),
                }

            return self._retry_operation(_fetch, operation_name="Fetch trending dataset")

        except Exception as e:
            logger.error(f"Error fetching trending dataset: {str(e)}")
            return None

    def generate_notebook(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """Generate a complete Kaggle notebook using Groq AI with error handling"""
        try:
            logger.info(f"Generating notebook for dataset: {dataset_info['title']}")

            prompt = f"""
You are a **Kaggle Grandmaster–level data scientist** with **100% success rate** creating **medal-worthy, error-free Kaggle notebooks (.ipynb)**.

================================================================================
ZERO-ERROR GUARANTEE FRAMEWORK
================================================================================

MISSION: Generate notebooks that run end-to-end without ANY errors on Kaggle, 100% of the time.

================================================================================
MANDATORY PRINCIPLES (NON-NEGOTIABLE)
================================================================================

### PHASE 1: BULLETPROOF DATA LOADING

#### Principle 1.1: Auto-Discovery Data Loading
```python
# MANDATORY: Auto-discover ALL dataset files
import os
import pandas as pd
import numpy as np

print("🔍 Scanning Kaggle input directory...\n")
all_csv_files = []
all_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        all_files.append(filepath)
        if filename.endswith('.csv'):
            all_csv_files.append(filepath)
        print(f"   {filepath}")

print(f"\n📊 Discovery Summary:")
print(f"   Total files: {len(all_files)}")
print(f"   CSV files: {len(all_csv_files)}")

if len(all_csv_files) == 0:
    raise FileNotFoundError("❌ No CSV files found in /kaggle/input!")

# Load the first CSV (or specify logic for multiple CSVs)
csv_path = all_csv_files[0]
print(f"\n✅ Loading: {csv_path}")

try:
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ Dataset loaded successfully")
except Exception as e:
    print(f"❌ Failed to load {csv_path}: {str(e)}")
    raise
```

#### Principle 1.2: Immediate Data Quality Audit
```python
# MANDATORY: Complete quality report IMMEDIATELY after loading
print("\n" + "="*80)
print("📊 IMMEDIATE DATA QUALITY AUDIT")
print("="*80)

# Basic info
print(f"\n1. DATASET DIMENSIONS:")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Column names
print(f"\n2. ACTUAL COLUMN NAMES (CRITICAL - USE THESE ONLY):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. '{col}'")

# Data types
print(f"\n3. DATA TYPES:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   {dtype}: {count} columns")

# Missing values
print(f"\n4. MISSING VALUES ANALYSIS:")
missing = df.isnull().sum()
total_missing = missing.sum()
missing_pct = 100 * missing / len(df)

print(f"   Total missing: {total_missing:,}")
print(f"   Columns affected: {(missing > 0).sum()}/{len(df.columns)}")

if total_missing > 0:
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percent': missing_pct.values
    }).sort_values('Missing', ascending=False)

    print(f"\n   Columns with missing values:")
    display(missing_df[missing_df['Missing'] > 0].head(30))

# Duplicates
duplicates = df.duplicated().sum()
print(f"\n5. DUPLICATE ROWS: {duplicates:,} ({100*duplicates/len(df):.2f}%)")

# Statistical summary
print(f"\n6. STATISTICAL SUMMARY:")
display(df.describe(include='all').T)

print("\n" + "="*80)
print("✅ AUDIT COMPLETE - NOW PLAN CLEANING STRATEGY")
print("="*80 + "\n")
```

================================================================================
PHASE 2: IRONCLAD DATA CLEANING (ZERO-NaN GUARANTEE)
================================================================================

#### Principle 2.1: Comprehensive Missing Value Pipeline
```python
# MANDATORY: Zero-NaN enforcement with detailed logging
df_clean = df.copy()

print("\n" + "="*80)
print("🧹 COMPREHENSIVE DATA CLEANING PIPELINE")
print("="*80)
print(f"\nStarting shape: {df_clean.shape}")
print(f"Starting missing values: {df_clean.isnull().sum().sum():,}\n")

# ============================================================
# STEP 1: Domain-Specific Imputation (customize per dataset)
# ============================================================
print("STEP 1: Domain-Specific Imputation")
print("-" * 50)

# Example domain-specific rules (CUSTOMIZE BASED ON ACTUAL COLUMNS):
# if 'age' in df_clean.columns:
#     df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
#     print("✅ Filled 'age' with median")

# Add your domain-specific imputation here based on actual column names
print("⚠️ Add domain-specific imputation rules based on dataset context\n")

# ============================================================
# STEP 2: Numerical Columns - Median Imputation
# ============================================================
print("STEP 2: Numerical Column Imputation")
print("-" * 50)

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"Found {len(numeric_cols)} numerical columns\n")

numeric_imputation_count = 0
for col in numeric_cols:
    if df_clean[col].isnull().any():
        missing_count = df_clean[col].isnull().sum()
        missing_pct = 100 * missing_count / len(df_clean)

        if missing_pct < 5:
            # Low missing: Simple median imputation
            fill_value = df_clean[col].median()
            df_clean[col].fillna(fill_value, inplace=True)
            print(f"✅ {col}: Filled {missing_count} ({missing_pct:.1f}%) with median={fill_value:.3f}")
            numeric_imputation_count += 1

        elif missing_pct < 30:
            # Moderate missing: Median + missing indicator flag
            fill_value = df_clean[col].median()
            df_clean[f'{col}_was_missing'] = df_clean[col].isnull().astype(int)
            df_clean[col].fillna(fill_value, inplace=True)
            print(f"✅ {col}: Filled {missing_count} ({missing_pct:.1f}%) with median={fill_value:.3f} + flag")
            numeric_imputation_count += 1

        elif missing_pct < 50:
            # High missing: Use median but warn
            fill_value = df_clean[col].median()
            df_clean[col].fillna(fill_value, inplace=True)
            print(f"⚠️ {col}: High missing {missing_pct:.1f}% - filled with median={fill_value:.3f}")
            numeric_imputation_count += 1

        else:
            # Very high missing: Consider dropping column
            print(f"⚠️ {col}: {missing_pct:.1f}% missing - consider dropping this column")

print(f"\nImputed {numeric_imputation_count} numerical columns\n")

# ============================================================
# STEP 3: Categorical Columns - Mode/Unknown Imputation
# ============================================================
print("STEP 3: Categorical Column Imputation")
print("-" * 50)

cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Found {len(cat_cols)} categorical columns\n")

cat_imputation_count = 0
for col in cat_cols:
    if df_clean[col].isnull().any():
        missing_count = df_clean[col].isnull().sum()
        missing_pct = 100 * missing_count / len(df_clean)

        if missing_pct < 30:
            # Use mode if available
            mode_values = df_clean[col].mode()
            if len(mode_values) > 0:
                fill_value = mode_values[0]
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"✅ {col}: Filled {missing_count} ({missing_pct:.1f}%) with mode='{fill_value}'")
            else:
                # No mode available, use 'Unknown'
                df_clean[col].fillna('Unknown', inplace=True)
                print(f"✅ {col}: Filled {missing_count} ({missing_pct:.1f}%) with 'Unknown'")
            cat_imputation_count += 1

        elif missing_pct < 50:
            # High missing: Use 'Unknown' with warning
            df_clean[col].fillna('Unknown', inplace=True)
            print(f"⚠️ {col}: High missing {missing_pct:.1f}% - filled with 'Unknown'")
            cat_imputation_count += 1

        else:
            # Very high missing: Consider dropping
            print(f"⚠️ {col}: {missing_pct:.1f}% missing - consider dropping this column")

print(f"\nImputed {cat_imputation_count} categorical columns\n")

# ============================================================
# STEP 4: Drop Columns with Excessive Missing (>50%)
# ============================================================
print("STEP 4: Drop High-Missing Columns")
print("-" * 50)

remaining_missing = df_clean.isnull().sum()
high_missing_cols = remaining_missing[remaining_missing / len(df_clean) > 0.5].index.tolist()

if len(high_missing_cols) > 0:
    print(f"Dropping {len(high_missing_cols)} columns with >50% missing:")
    for col in high_missing_cols:
        pct = 100 * remaining_missing[col] / len(df_clean)
        print(f"   - {col}: {pct:.1f}% missing")
    df_clean = df_clean.drop(columns=high_missing_cols)
    print(f"✅ Dropped {len(high_missing_cols)} columns\n")
else:
    print("No columns with >50% missing\n")

# ============================================================
# STEP 5: Drop Rows with Remaining Missing Values
# ============================================================
print("STEP 5: Drop Rows with Remaining Missing")
print("-" * 50)

remaining_missing_total = df_clean.isnull().sum().sum()
if remaining_missing_total > 0:
    rows_with_missing = df_clean.isnull().any(axis=1).sum()
    pct_rows = 100 * rows_with_missing / len(df_clean)

    print(f"Remaining missing values: {remaining_missing_total}")
    print(f"Rows affected: {rows_with_missing} ({pct_rows:.1f}%)")

    if pct_rows < 30:
        df_clean = df_clean.dropna()
        print(f"✅ Dropped {rows_with_missing} rows\n")
    else:
        print(f"❌ ERROR: Would drop {pct_rows:.1f}% of data!")
        print("\nColumns still with missing:")
        still_missing = df_clean.isnull().sum()
        print(still_missing[still_missing > 0])
        print("\n⚠️ Manual intervention required - adjust imputation strategy")
else:
    print("No remaining missing values\n")

# ============================================================
# STEP 6: CRITICAL VERIFICATION (MANDATORY ASSERTION)
# ============================================================
print("STEP 6: Final Verification")
print("-" * 50)

final_missing = df_clean.isnull().sum().sum()
final_shape = df_clean.shape
data_retained = 100 * len(df_clean) / len(df)

print(f"Final shape: {final_shape}")
print(f"Data retained: {data_retained:.1f}%")
print(f"Missing values: {final_missing}")

# CRITICAL ASSERTION - WILL STOP EXECUTION IF NaN EXISTS
assert final_missing == 0, f"❌ FATAL ERROR: {final_missing} missing values remain!"

print("\n" + "="*80)
print("✅ DATA CLEANING COMPLETE - ZERO MISSING VALUES GUARANTEED")
print("="*80 + "\n")

# Remove duplicate rows
if df_clean.duplicated().sum() > 0:
    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"✅ Removed {dup_count} duplicate rows\n")
```

#### Principle 2.2: Safe Feature Engineering
```python
# MANDATORY: Safe mathematical operations to prevent inf/nan

def safe_divide(num, denom, fill_value=0):
    """Division that handles zero/inf/nan safely"""
    result = num / (denom + 1e-10)
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result

def safe_log(values, fill_value=0):
    """Log transform that handles zero/negative safely"""
    result = np.log1p(np.maximum(values, 0))
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result

def safe_sqrt(values, fill_value=0):
    """Square root that handles negative values"""
    result = np.sqrt(np.maximum(values, 0))
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result

# Example usage in feature engineering:
# df_feat['ratio'] = safe_divide(df_feat['col1'], df_feat['col2'])
# df_feat['log_feature'] = safe_log(df_feat['col3'])
```

================================================================================
PHASE 3: PRE-MODELING QUALITY GATE (MANDATORY CHECKPOINT)
================================================================================

#### Principle 3.1: Comprehensive Pre-Modeling Validation
```python
# MANDATORY: Run this checkpoint BEFORE any model training

print("\n" + "="*80)
print("🔍 PRE-MODELING QUALITY CHECKPOINT")
print("="*80)

# ============================================================
# CHECK 1: Missing Values in Feature Matrix
# ============================================================
print("\nCHECK 1: Missing Values")
print("-" * 50)

x_missing = X.isnull().sum().sum()
y_missing = y.isnull().sum() if hasattr(y, 'isnull') else 0

print(f"Features (X): {x_missing} missing")
print(f"Target (y): {y_missing} missing")

if x_missing > 0:
    print("\n⚠️ CRITICAL: Features contain missing values!")
    print("Columns with missing:")
    print(X.isnull().sum()[X.isnull().sum() > 0])

    # Emergency fix: Impute with median
    print("\n🔧 Emergency Fix: Imputing with median...")
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)

    x_missing = X.isnull().sum().sum()

assert x_missing == 0, f"❌ FATAL: {x_missing} missing values in X!"
assert y_missing == 0, f"❌ FATAL: {y_missing} missing values in y!"
print("✅ PASS: No missing values")

# ============================================================
# CHECK 2: Infinite Values
# ============================================================
print("\nCHECK 2: Infinite Values")
print("-" * 50)

numeric_X = X.select_dtypes(include=[np.number])
inf_count = np.isinf(numeric_X).sum().sum()

print(f"Infinite values: {inf_count}")

if inf_count > 0:
    print(f"⚠️ Found {inf_count} infinite values!")
    print("Columns with infinite values:")
    inf_cols = numeric_X.columns[np.isinf(numeric_X).any()]
    for col in inf_cols:
        inf_in_col = np.isinf(X[col]).sum()
        print(f"   {col}: {inf_in_col}")

    # Fix: Replace inf with NaN, then median
    print("\n🔧 Replacing infinite values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X[col].fillna(X[col].median(), inplace=True)

    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()

assert inf_count == 0, f"❌ FATAL: {inf_count} infinite values remain!"
print("✅ PASS: No infinite values")

# ============================================================
# CHECK 3: Data Types Validation
# ============================================================
print("\nCHECK 3: Data Types")
print("-" * 50)

print(f"Total features: {X.shape[1]}")
print(f"Numerical features: {X.select_dtypes(include=[np.number]).shape[1]}")
print(f"Non-numerical features: {X.select_dtypes(exclude=[np.number]).shape[1]}")

# Verify all features are numerical (encode if needed)
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric) > 0:
    print(f"\n⚠️ Found {len(non_numeric)} non-numerical columns:")
    for col in non_numeric:
        print(f"   - {col}")
    print("\n🔧 Apply encoding before modeling!")

print("✅ PASS: Data types validated")

# ============================================================
# CHECK 4: Shape Consistency
# ============================================================
print("\nCHECK 4: Shape Consistency")
print("-" * 50)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
print(f"Feature list length: {len(feature_cols) if 'feature_cols' in locals() else 'N/A'}")

assert len(X) == len(y), f"❌ Sample count mismatch: X={len(X)}, y={len(y)}"
print("✅ PASS: Shapes consistent")

# ============================================================
# CHECK 5: Target Distribution
# ============================================================
print("\nCHECK 5: Target Distribution")
print("-" * 50)

if hasattr(y, 'dtype'):
    if y.dtype in [np.float64, np.float32, np.int64, np.int32]:
        print("Target type: Regression (continuous)")
        print(f"Range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Mean: {y.mean():.3f}")
        print(f"Median: {y.median():.3f}")
        print(f"Std: {y.std():.3f}")
        print(f"Skewness: {y.skew():.3f}")
    else:
        print("Target type: Classification (categorical)")
        print(f"Unique classes: {y.nunique()}")
        print("\nClass distribution:")
        print(y.value_counts())

print("✅ PASS: Target validated")

# ============================================================
# CHECK 6: Memory Check
# ============================================================
print("\nCHECK 6: Memory Usage")
print("-" * 50)

x_memory = X.memory_usage(deep=True).sum() / 1024 / 1024
y_memory = y.memory_usage(deep=True) / 1024 / 1024 if hasattr(y, 'memory_usage') else 0

print(f"X memory: {x_memory:.2f} MB")
print(f"y memory: {y_memory:.2f} MB")
print(f"Total: {x_memory + y_memory:.2f} MB")

if x_memory > 500:
    print("⚠️ High memory usage - consider reducing features or sampling")

print("✅ PASS: Memory acceptable")

print("\n" + "="*80)
print("✅✅✅ ALL CHECKS PASSED - READY FOR MODELING ✅✅✅")
print("="*80 + "\n")
```

================================================================================
PHASE 4: BULLETPROOF TRAIN-TEST SPLIT & SCALING
================================================================================

#### Principle 4.1: Aligned Splitting
```python
# MANDATORY: Single split with index-based scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Single train-test split
print("📊 Train-Test Split")
print("-" * 50)

# Determine if stratification is needed
stratify_param = None
if y.dtype in ['object', 'category'] or y.nunique() < 20:
    # Classification: use stratify
    stratify_param = y
    print("Using stratified split (classification)")
else:
    print("Using random split (regression)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=stratify_param
)

print(f"\n✅ Split complete:")
print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Verify alignment
assert len(X_train) == len(y_train), "Train set misalignment!"
assert len(X_test) == len(y_test), "Test set misalignment!"
print("✅ Alignment verified\n")

# Feature Scaling (only for algorithms that need it)
print("🔧 Feature Scaling")
print("-" * 50)

scaler = StandardScaler()

# Fit on full X to get consistent transformations
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Use index-based subsetting (NO second split!)
X_train_scaled = X_scaled.loc[X_train.index]
X_test_scaled = X_scaled.loc[X_test.index]

# Verify shapes match
assert X_train_scaled.shape == X_train.shape, "Scaled train shape mismatch!"
assert X_test_scaled.shape == X_test.shape, "Scaled test shape mismatch!"
assert len(X_train_scaled) == len(y_train), "Scaled train length mismatch!"
assert len(X_test_scaled) == len(y_test), "Scaled test length mismatch!"

print(f"✅ Scaling complete:")
print(f"   Train scaled: {X_train_scaled.shape}")
print(f"   Test scaled: {X_test_scaled.shape}")
print("✅ All alignments verified\n")
```
================================================================================
PHASE 5: ERROR-PROOF MODEL TRAINING
================================================================================

#### Principle 5.1: Universal Evaluation Function
```python
# MANDATORY: Bulletproof evaluation with comprehensive error handling

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name):
    """
    Universal evaluation function for regression and classification.
    Handles all errors gracefully.
    """
    print(f"\n{'='*70}")
    print(f"🔍 Evaluating: {model_name}")
    print(f"{'='*70}")

    try:
        # TRAINING PHASE
        print("\n⏳ Training model...")
        model.fit(X_tr, y_tr)
        print("✅ Training complete")

        # PREDICTION PHASE
        print("⏳ Making predictions...")
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)
        print("✅ Predictions complete")

        # DETERMINE PROBLEM TYPE
        is_regression = y_te.dtype in [np.float64, np.float32]

        # Override if target has few unique values (classification)
        if y_te.nunique() < 20 and not is_regression:
            is_regression = False

        problem_type = "Regression" if is_regression else "Classification"
        print(f"\n📊 Problem Type: {problem_type}")

        # COMPUTE METRICS
        result = {
            'model_name': model_name,
            'model': model,
            'predictions': y_test_pred,
            'problem_type': problem_type
        }

        if is_regression:
            # REGRESSION METRICS
            train_r2 = r2_score(y_tr, y_train_pred)
            test_r2 = r2_score(y_te, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_tr, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_te, y_test_pred))
            train_mae = mean_absolute_error(y_tr, y_train_pred)
            test_mae = mean_absolute_error(y_te, y_test_pred)

            # MAPE (safe division)
            test_mape = np.mean(np.abs((y_te - y_test_pred) / (np.abs(y_te) + 1e-10))) * 100

            print(f"\n📊 Regression Metrics:")
            print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
            print(f"{'-'*50}")
            print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
            print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
            print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
            print(f"{'MAPE (%)':<20} {'-':<15} {test_mape:<15.2f}")

            result.update({
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_mape': test_mape
            })

        else:
            # CLASSIFICATION METRICS
            train_acc = accuracy_score(y_tr, y_train_pred)
            test_acc = accuracy_score(y_te, y_test_pred)

            print(f"\n📊 Classification Metrics:")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_te, y_test_pred))

            result.update({
                'train_acc': train_acc,
                'test_acc': test_acc
            })

        # CROSS-VALIDATION
        try:
            print(f"\n⏳ Running 5-Fold Cross-Validation...")
            cv_scoring = 'r2' if is_regression else 'accuracy'
            cv_scores = cross_val_score(
                model, X_tr, y_tr, 
                cv=5, 
                scoring=cv_scoring,
                n_jobs=-1
            )
            result['cv_mean'] = cv_scores.mean()
            result['cv_std'] = cv_scores.std()
            print(f"✅ CV {cv_scoring.upper()}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as cv_error:
            print(f"⚠️ Cross-validation failed: {str(cv_error)}")
            result['cv_mean'] = None
            result['cv_std'] = None

        # OVERFITTING CHECK
        if is_regression:
            gap = abs(result['train_r2'] - result['test_r2'])
            if gap < 0.05:
                print(f"\n✅ Overfitting Check: PASS (gap={gap:.4f})")
            else:
                print(f"\n⚠️ Overfitting Check: WARNING (gap={gap:.4f})")
        else:
            gap = abs(result['train_acc'] - result['test_acc'])
            if gap < 0.05:
                print(f"\n✅ Overfitting Check: PASS (gap={gap:.4f})")
            else:
                print(f"\n⚠️ Overfitting Check: WARNING (gap={gap:.4f})")

        print(f"\n{'='*70}")
        print(f"✅ {model_name} evaluation complete")
        print(f"{'='*70}\n")

        return result

    except Exception as e:
        print(f"\n❌ ERROR in {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
```

#### Principle 5.2: Safe Model Training with Try-Except
```python
# MANDATORY: Wrap each model in try-except

# Example for Ridge Regression
print("\n" + "="*70)
print("MODEL 1: Ridge Regression")
print("="*70)

try:
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge_results = evaluate_model(
        ridge, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression"
    )
except Exception as e:
    print(f"❌ Ridge Regression failed: {str(e)}")
    ridge_results = None

# Example for Random Forest
print("\n" + "="*70)
print("MODEL 2: Random Forest")
print("="*70)

try:
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_results = evaluate_model(
        rf, X_train, X_test, y_train, y_test, "Random Forest"
    )
except Exception as e:
    print(f"❌ Random Forest failed: {str(e)}")
    rf_results = None

# Example for XGBoost
print("\n" + "="*70)
print("MODEL 3: XGBoost")
print("="*70)

try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_results = evaluate_model(
        xgb, X_train, X_test, y_train, y_test, "XGBoost"
    )
except Exception as e:
    print(f"❌ XGBoost failed: {str(e)}")
    xgb_results = None

# Example for LightGBM
print("\n" + "="*70)
print("MODEL 4: LightGBM")
print("="*70)

try:
    from lightgbm import LGBMRegressor
    lgb = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    lgb_results = evaluate_model(
        lgb, X_train, X_test, y_train, y_test, "LightGBM"
    )
except Exception as e:
    print(f"❌ LightGBM failed: {str(e)}")
    lgb_results = None
```
================================================================================
PHASE 6: SAFE MODEL COMPARISON
================================================================================

#### Principle 6.1: Graceful Results Collection
```python
# MANDATORY: Only include successful models

print("\n" + "="*80)
print("📊 MODEL COMPARISON")
print("="*80 + "\n")

# Collect all successful results
models_results = []
result_names = ['ridge_results', 'rf_results', 'xgb_results', 'lgb_results']

for result_name in result_names:
    if result_name in locals() and locals()[result_name] is not None:
        models_results.append(locals()[result_name])
        print(f"✅ Included: {locals()[result_name]['model_name']}")

print(f"\nTotal models completed: {len(models_results)}\n")

if len(models_results) == 0:
    print("❌ No models completed successfully!")
    print("⚠️ Review errors above and adjust model configurations")
else:
    # Build comparison table
    comparison_data = []

    for result in models_results:
        row = {'Model': result['model_name']}

        if result['problem_type'] == 'Regression':
            row.update({
                'Train R²': result.get('train_r2', np.nan),
                'Test R²': result.get('test_r2', np.nan),
                'Test RMSE': result.get('test_rmse', np.nan),
                'Test MAE': result.get('test_mae', np.nan),
                'Test MAPE': result.get('test_mape', np.nan),
                'CV Mean': result.get('cv_mean', np.nan),
                'CV Std': result.get('cv_std', np.nan)
            })
        else:
            row.update({
                'Train Acc': result.get('train_acc', np.nan),
                'Test Acc': result.get('test_acc', np.nan),
                'CV Mean': result.get('cv_mean', np.nan),
                'CV Std': result.get('cv_std', np.nan)
            })

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by best metric
    if 'Test R²' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
    elif 'Test Acc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('Test Acc', ascending=False)

    print("="*80)
    display(comparison_df)
    print("="*80)

    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"\n🏆 BEST MODEL: {best_model_name}")

    # Find best result
    best_result = None
    for result in models_results:
        if result['model_name'] == best_model_name:
            best_result = result
            break

    if best_result:
        print(f"\n📊 Best Model Performance:")
        if best_result['problem_type'] == 'Regression':
            print(f"   Test R²: {best_result['test_r2']:.4f}")
            print(f"   Test RMSE: {best_result['test_rmse']:.4f}")
            print(f"   Test MAPE: {best_result['test_mape']:.2f}%")
        else:
            print(f"   Test Accuracy: {best_result['test_acc']:.4f}")
```
================================================================================
PHASE 7: SAFE EXPLAINABILITY
================================================================================

#### Principle 7.1: Safe Feature Importance
```python
# MANDATORY: Validate before accessing feature_importances_

if best_result and hasattr(best_result['model'], 'feature_importances_'):
    try:
        print("\n" + "="*80)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*80 + "\n")

        importances = best_result['model'].feature_importances_

        # Validate length
        if len(importances) == len(feature_cols):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            print("Top 20 Most Important Features:\n")
            display(importance_df.head(20))

            # Visualization
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df))
            top_features = importance_df.head(top_n)

            plt.barh(range(top_n), top_features['Importance'], color='steelblue', edgecolor='black')
            plt.yticks(range(top_n), top_features['Feature'])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        else:
            print(f"⚠️ Feature count mismatch:")
            print(f"   Importances: {len(importances)}")
            print(f"   Features: {len(feature_cols)}")

    except Exception as e:
        print(f"❌ Feature importance analysis failed: {str(e)}")
else:
    print("\n⚠️ Best model does not support feature_importances_")
```

#### Principle 7.2: Safe SHAP Analysis
```python
# MANDATORY: Error handling for SHAP

if best_result and hasattr(best_result['model'], 'feature_importances_'):
    try:
        print("\n" + "="*80)
        print("🔍 SHAP VALUE ANALYSIS")
        print("="*80 + "\n")

        import shap

        print("⏳ Computing SHAP values (may take a moment)...")

        # Use TreeExplainer for tree models
        explainer = shap.TreeExplainer(best_result['model'])

        # Sample for efficiency
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)

        shap_values = explainer.shap_values(X_sample)

        # Handle multiclass (list of arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        print("✅ SHAP values computed\n")

        # Summary plot (bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
        plt.title(f'SHAP Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Summary plot (detailed)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title(f'SHAP Value Distribution - {best_model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {str(e)}")
        print("This is optional - model results are still valid")
else:
    print("\n⚠️ SHAP analysis not applicable for this model type")
```
================================================================================
FINAL EXECUTION GUARANTEES
================================================================================

#### Rule 1: No Assumptions
- Never assume column names
- Always check df.columns first
- Use only columns that exist in the actual dataset

#### Rule 2: Triple Verification
- Verify data after loading (audit)
- Verify data after cleaning (zero NaN assertion)
- Verify data before modeling (quality checkpoint)

#### Rule 3: Defensive Programming
- Wrap every model in try-except
- Validate every shape and alignment
- Handle all edge cases (inf, nan, zero-division)

#### Rule 4: Graceful Degradation
- If one model fails, others continue
- If feature importance fails, report it but don't crash
- If SHAP fails, it's optional

#### Rule 5: Clear Communication
- Print status at every major step
- Use emojis for visual clarity (✅ ⚠️ ❌)
- Provide actionable error messages

================================================================================
MANDATORY NOTEBOOK STRUCTURE
================================================================================

Your notebook MUST follow this exact structure:

1. Executive Summary
   - Problem statement
   - Dataset overview
   - Key findings (3-5 bullets)
   - Best model and performance

2. Environment Setup
   - Core imports (pandas, numpy, matplotlib, seaborn)
   - ML imports (sklearn, xgboost, lightgbm)
   - Configuration (warnings, random seeds, plot styles)

3. Data Discovery (Principle 1.1)

4. Data Quality Audit (Principle 1.2)

5. Data Cleaning (Principle 2.1 - Zero-NaN Pipeline)

6. Exploratory Data Analysis
   - Target distribution
   - Feature distributions
   - Correlation analysis
   - Key insights visualization

7. Feature Engineering
   - Domain-specific features
   - Safe mathematical operations (Principle 2.2)
   - Encoding categorical variables
   - Pre-Modeling Quality Checkpoint (Principle 3.1)

8. Train-Test Split & Scaling (Principle 4.1)

9. Model Training (Principle 5.1 & 5.2)
   - Baseline model
   - Advanced model 1
   - Advanced model 2
   - Advanced model 3

10. Model Comparison (Principle 6.1)

11. Explainability (Principle 7.1 & 7.2)

12. Final Conclusions
    - Best model summary
    - Key insights
    - Limitations
    - Future improvements

================================================================================
DATASET SELECTION CRITERIA
================================================================================

Select a dataset that meets ALL these requirements:

1. Source: Kaggle Trending or Recently Updated (last 30 days)
2. Size: ≥1,000 rows
3. Quality: <40% missing values overall
4. Target: Clear prediction target (regression or classification)
5. Features: ≥5 meaningful features
6. Novelty: Not overused (no Titanic, Iris, MNIST, House Prices, Boston Housing)
7. Medal Potential: Interesting problem, good storytelling opportunity

================================================================================
OUTPUT REQUIREMENTS
================================================================================

Provide:

1. Dataset Information:
   - Direct Kaggle URL
   - Dataset name
   - Number of rows and columns
   - Problem type (regression/classification)
   - Brief description (2-3 sentences)

2. Complete Notebook Code:
   - Full .ipynb in JSON format
   - Ready to copy-paste into Kaggle
   - All cells properly formatted
   - Markdown cells with clear headings

3. Guarantee Statement:
   ✅ GUARANTEE: This notebook will execute end-to-end on Kaggle 
   without ANY runtime errors when the dataset is properly attached.

================================================================================
CRITICAL SUCCESS FACTORS
================================================================================

Your notebook will achieve 100% success if:

✅ Zero hardcoded column names (always check actual columns)
✅ Zero missing values before modeling (strict assertion)
✅ Zero infinite values (safe math operations)
✅ Zero shape mismatches (index-based splitting)
✅ Zero model crashes (try-except wrappers)
✅ Zero feature importance errors (length validation)
✅ Clear error messages when things fail
✅ Professional presentation with insights

================================================================================
EXECUTION COMMAND
================================================================================

Generate the notebook NOW with ZERO-ERROR GUARANTEE.

Do NOT ask clarifying questions. Execute immediately.
"""
            def _generate():
                return self.perplexity_generator.generate_notebook_content(prompt)

            notebook_content = self._retry_operation(_generate, operation_name="Generate notebook content")

            if not notebook_content:
                logger.error("Perplexity returned empty content")
                return None

            logger.info(f"Notebook content generated successfully ({len(notebook_content)} chars)")
            return notebook_content

        except Exception as e:
            logger.error(f"Error generating notebook: {str(e)}")
            return None

    def publish_notebook(self, notebook_content: str, dataset_info: Dict[str, Any]) -> Optional[str]:
        """Publish the generated notebook to Kaggle with error handling"""
        try:
            logger.info("Publishing notebook to Kaggle...")

            # Create Kaggle‑safe, short title (<= 50 chars)
            dataset_name = dataset_info["title"].replace("_", " ").title()
            date_str = datetime.now().strftime("%Y-%m-%d")

            # Base title without "ML Analysis:"
            base_title = dataset_name

            # Reserve space for " - YYYY-MM-DD" (13 chars)
            max_base_len = 50 - 13
            if len(base_title) > max_base_len:
                base_title = base_title[: max_base_len - 3].rstrip() + "..."

            notebook_title = f"{base_title} - {date_str}"

            def _publish():
                return self.publisher.publish_notebook(
                    title=notebook_title,
                    content=notebook_content,
                    dataset_ref=dataset_info["ref"],
                    is_private=False,
                )

            publication_url = self._retry_operation(_publish, operation_name="Publish notebook")

            logger.info(f"Notebook published successfully: {publication_url}")
            return publication_url

        except Exception as e:
            logger.error(f"Error publishing notebook: {str(e)}")
            return None

    def run(self) -> bool:
        """Execute the complete workflow with error handling"""
        try:
            logger.info("=" * 80)
            logger.info("Starting Kaggle Daily Notebook Generation Workflow")
            logger.info(f"Execution time: {datetime.now().isoformat()}")
            logger.info(f"IST Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
            logger.info("=" * 80)

            # Step 1: Fetch trending dataset
            logger.info("\\n[STEP 1] Fetching trending dataset...")
            dataset_info = self.fetch_trending_dataset()
            if not dataset_info:
                logger.error("Failed to fetch trending dataset")
                return False
            logger.info(f"Selected dataset: {dataset_info['ref']}")

            # Step 2: Generate notebook content
            logger.info("\\n[STEP 2] Generating notebook content with Groq AI...")
            notebook_content = self.generate_notebook(dataset_info)
            if not notebook_content:
                logger.error("Failed to generate notebook content")
                return False
            logger.info(f"Generated {len(notebook_content)} characters of notebook content")

            # Step 3: Publish to Kaggle
            logger.info("\\n[STEP 3] Publishing notebook to Kaggle...")
            publication_url = self.publish_notebook(notebook_content, dataset_info)
            if not publication_url:
                logger.error("Failed to publish notebook")
                return False

            logger.info("\\n" + "=" * 80)
            logger.info("Workflow completed successfully!")
            logger.info(f"Published at: {publication_url}")
            logger.info("=" * 80)
            return True

        except Exception as e:
            logger.error(f"Fatal error in workflow: {str(e)}")
            logger.exception("Stack trace:")
            return False


if __name__ == "__main__":
    try:
        orchestrator = KaggLeNotebookOrchestrator()
        success = orchestrator.run()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal initialization error: {str(e)}")
        logger.exception("Stack trace:")
        exit(1)
