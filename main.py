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

The dataset you must analyze is: **'{dataset_info['title']}'**.

================================================================================
ZERO-ERROR GUARANTEE FRAMEWORK
================================================================================

MISSION: Generate notebooks that run end-to-end without ANY errors on Kaggle, 100% of the time.

You MUST:
- Auto-discover dataset files under `/kaggle/input`.
- Load the correct CSV(s) robustly.
- Perform a full data audit, cleaning, feature engineering, modeling, evaluation, and explainability.
- Follow the coding patterns exactly as shown in the template blocks below (they are reference patterns, not literal strings).

STRICT OUTPUT FORMAT:
- Output ONLY a valid Jupyter notebook in JSON (nbformat 4). No backticks, no markdown fences, no extra prose.
- Use multiple markdown and code cells, not one giant cell.
- Use clear headings: #, ##, ### with blank lines after headings.
- Code must be PEP8-style, one statement per line (no `;`), logically split into cells.

================================================================================
PHASE 1: BULLETPROOF DATA LOADING
================================================================================

You MUST implement:
- Auto-discovery of all files under `/kaggle/input`.
- Selection and loading of the primary CSV into `df`.
- Immediate data quality audit (shape, dtypes, missing, duplicates, describe).

Follow the patterns shown below, but USE ACTUAL COLUMN NAMES from this dataset:

(Example pattern – adapt, do not hardcode)

- Scan `/kaggle/input` and list CSVs.
- Load first CSV or a clearly justified main CSV.
- Print audit: dims, dtypes, missing summary, duplicates, describe.

================================================================================
PHASE 2: IRONCLAD DATA CLEANING (ZERO-NaN GUARANTEE)
================================================================================

You MUST:
- Create `df_clean` as a cleaned copy of `df`.
- Implement:
  - Domain-specific imputations where obvious from column names.
  - Numeric imputation with median + optional missing flags.
  - Categorical imputation with mode/"Unknown".
  - Drop columns with >50% missing if needed.
  - Drop rows with remaining missing only if <30% of rows, otherwise adjust strategy.
- Assert at the end that `df_clean` has ZERO missing values and no duplicates.

Implement helper utilities for safe transforms:

- `safe_divide(num, denom, fill_value=0)`
- `safe_log(values, fill_value=0)`
- `safe_sqrt(values, fill_value=0)`

These must:
- Avoid division by zero.
- Replace inf/-inf and NaN with `fill_value`.

Use them in feature engineering where appropriate.

================================================================================
PHASE 3: PRE-MODELING QUALITY GATE
================================================================================

You MUST:
- Build feature matrix `X` and target `y` based on the problem (classification vs regression).
- Run a pre-modeling checkpoint that:
  - Checks and fixes any remaining NaN/inf in X/y (with clear logs).
  - Confirms all features in X are numeric (encode categoricals if needed).
  - Verifies shapes and alignment of X, y, and feature list.
  - Prints target distribution (stats for regression, class counts for classification).
  - Prints memory usage of X and y and warns if too large.

================================================================================
PHASE 4: BULLETPROOF TRAIN-TEST SPLIT & SCALING
================================================================================

You MUST:
- Use a single `train_test_split` with `test_size=0.2` and `random_state=42`.
- Use `stratify=y` for classification (few classes), no stratify for regression.
- Apply scaling via `StandardScaler`:
  - Fit scaler on full X to get consistent transformation.
  - Create `X_scaled` as DataFrame.
  - Derive `X_train_scaled` and `X_test_scaled` by indexing on train/test indices.
- Assert shapes and alignment after scaling.

================================================================================
PHASE 5: ERROR-PROOF MODEL TRAINING
================================================================================

You MUST implement:

- A universal `evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name)` that:
  - Fits model, predicts, determines regression vs classification.
  - For regression: R², RMSE, MAE, MAPE, optional CV R² (5-fold).
  - For classification: accuracy, classification report, optional CV accuracy.
  - Performs an overfitting check by comparing train vs test metrics.
  - Wraps everything in try/except and prints detailed errors without crashing the notebook.

Train at least 3–4 models where applicable, e.g.:

- Ridge / LinearRegression.
- RandomForestRegressor or RandomForestClassifier.
- XGBRegressor / XGBClassifier (if available).
- LGBMRegressor / LGBMClassifier (if available).

Each model must be wrapped in try/except; failure of one model must not crash the notebook.

================================================================================
PHASE 6: SAFE MODEL COMPARISON AND BEST MODEL SELECTION
================================================================================

You MUST:
- Collect all successful `evaluate_model` results.
- Build a comparison DataFrame (Regression: Test R²/RMSE/MAE/MAPE; Classification: Train/Test accuracy, CV).
- Sort by best primary metric (Test R² or Test Accuracy).
- Print the best model name and its key metrics.

================================================================================
PHASE 7: SAFE EXPLAINABILITY
================================================================================

If the best model supports `feature_importances_`:

- Build and display a feature importance table (top 20 features).
- Plot a horizontal bar chart of top importances.

If SHAP is available and appropriate:

- Attempt SHAP TreeExplainer for tree models.
- Sample at most 100 rows from X_test.
- Generate SHAP summary plots (bar + swarm) with try/except; on failure, log a warning and continue.

================================================================================
NOTEBOOK STRUCTURE (CELLS)
================================================================================

Organize the notebook into clean sections:

1. Intro & Context (markdown)
2. Setup & Imports (code)
3. Data Loading & Audit (code + markdown)
4. Data Cleaning Pipeline (code + markdown)
5. Feature Engineering (code + markdown)
6. Pre-Modeling Checkpoint (code)
7. Train/Test Split & Scaling (code)
8. Modeling & Evaluation (multiple code + markdown)
9. Model Comparison & Best Model Summary (code + markdown)
10. Explainability (Feature Importance + SHAP if applicable)
11. Final Insights & Recommendations (markdown)

GENERAL PRESENTATION RULES (MANDATORY):
- Always put a blank line after a heading before text.
- Keep code cells between ~20–40 lines; split longer workflows.
- Use one statement per line; do NOT use `;` to join statements.
- Use clear, concise comments; avoid excessive decoration.
- Use consistent variable names across the notebook.

Notebook style should be similar to a top Kaggle EDA+model notebook:
- Catchy title with emoji.
- Quick summary bullets (dataset size, target, best model idea, key features).
- Sections: Setup, Fast Data Loading, EDA, Feature Engineering, Modeling, Evaluation, Insights.

Content requirements:
- Suggest which column is the target and why.
- Describe 3–5 most important features and why they matter.
- Provide clean EDA plots ideas (distributions, correlations, target vs features).
- Propose 2–3 models (one tree-based) with evaluation metrics.
- Add high-level narrative and conclusions.

Formatting rules:
- Use headings (#, ##, ###) with blank lines.
- Use bullet lists where appropriate.
- Wrap code in ```python fences.
- Keep code idiomatic but not too long; split logically.

Generate nicely formatted markdown + code blocks now.
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

            base_title = dataset_name
            max_base_len = 50 - 13  # space for " - YYYY-MM-DD"
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
