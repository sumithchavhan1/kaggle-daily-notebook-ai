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


class KaggleNotebookOrchestrator:
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

            prompt = f'''
You are a Kaggle Grandmaster–level data scientist with a proven track record of creating medal‑winning, error‑free Kaggle notebooks.

The dataset you must analyze is: "{dataset_info['title']}".

Your task:
- Produce a full end‑to‑end analysis and modeling notebook.
- The notebook must run without errors on Kaggle.
- Focus on clarity, teaching value, and strong modeling practices.

IMPORTANT OUTPUT FORMAT:
- Return ONLY markdown text and ```python code``` blocks.
- Do NOT return JSON or nbformat structures.
- Use multiple markdown and code cells (not one giant cell).
- Use clear headings: #, ##, ### with a blank line after each heading.

===============================================================================
NOTEBOOK STRUCTURE (SECTIONS)
===============================================================================

Aim for a structure similar to top Kaggle EDA+model notebooks:

1. Intro & Problem Summary (markdown)
2. Setup & Imports (code)
3. Data Loading & Audit (code + markdown)
4. Data Cleaning Pipeline (code + markdown)
5. Feature Engineering (code + markdown)
6. Pre‑Modeling Checkpoint (code)
7. Train/Test Split & Scaling (code)
8. Modeling & Evaluation (multiple code + markdown)
9. Model Comparison & Best Model Summary (code + markdown)
10. Explainability (Feature Importance + optional SHAP)
11. Final Insights & Recommendations (markdown)

You can add extra cells where useful; do not force fixed counts.

===============================================================================
PHASE 1: ROBUST DATA LOADING
===============================================================================

In the "Data Loading & Audit" section you MUST:

- Use os.walk("/kaggle/input") to find all .csv files recursively.
- Print all discovered CSV file paths in a numbered list.
- Choose the main CSV as the LARGEST file by size.
- Load it into a DataFrame named df with low_memory=False.
- After loading, print shape and columns.

Use this pattern (you may adjust variable names slightly, but preserve the logic):

```python
import os
import pandas as pd

input_dir = "/kaggle/input"
csv_files = []

for root, _, files in os.walk(input_dir):
    for f in files:
        if f.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, f))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {{input_dir}}")

print("📂 Discovered CSV files:")
for i, path in enumerate(csv_files, 1):
    print(f"{{i}}. {{path}}")

csv_files_sorted = sorted(csv_files, key=os.path.getsize, reverse=True)
main_csv_path = csv_files_sorted[0]

print(f"\\n✅ Loading main file: {{main_csv_path}}")
df = pd.read_csv(main_csv_path, low_memory=False)

print("\\n✅ Data loaded!")
print(f"Shape: {{df.shape}}")
print("Columns:", list(df.columns))
```

Immediately after loading, run a quick audit:

- df.head()
- df.info()
- df.describe(include="all").T
- Missing values summary (per column)
- Duplicate count

===============================================================================
PHASE 2: DATA CLEANING & FEATURE ENGINEERING
===============================================================================

Create df_clean as a cleaned copy of df.

Handle missing values for numeric and categorical columns.

Drop or flag columns with extreme missingness (>50%).

Remove duplicate rows if any.

Create a few meaningful features based on dataset context (date parts, ratios, interactions, etc.).

Ensure df_clean has no remaining NaN before modeling.

You may define helpers like:

safe_divide

safe_log

safe_sqrt

They must avoid division by zero and replace inf/NaN with safe values.

===============================================================================
PHASE 3: PRE‑MODELING QUALITY CHECKPOINT
===============================================================================

Before training models:

Define X (features) and y (target) based on dataset semantics; explain your choice.

Check:

No missing values in X or y (fix if any).

No inf/-inf in numeric columns.

All features in X are numeric (encode categoricals if needed).

Shapes and alignment: len(X) == len(y).

Print target distribution:

For regression: range, mean, median, std, skew.

For classification: class counts and percentages.

===============================================================================
PHASE 4: TRAIN/TEST SPLIT & SCALING
===============================================================================

Use train_test_split with test_size=0.2 and random_state=42.

For classification with few classes, use stratify=y; for regression, no stratify.

Scale features with StandardScaler:

Fit on full X to get consistent transform.

Create X_scaled as a DataFrame.

Derive X_train_scaled and X_test_scaled using train/test indices.

Assert shapes and alignment after scaling.

===============================================================================
PHASE 5: MODELING & EVALUATION
===============================================================================

Implement a universal helper:

evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name)

It should:

Fit the model.

Predict on train and test.

Detect regression vs classification.

For regression: R², RMSE, MAE, MAPE; optional 5‑fold CV R².

For classification: train/test accuracy, classification report; optional 5‑fold CV accuracy.

Print a compact, readable metric table.

Check for overfitting by comparing train vs test metrics.

Train at least 3 models where appropriate, for example:

LinearRegression or Ridge/Lasso.

RandomForestRegressor / RandomForestClassifier.

Gradient Boosting / XGBoost / LightGBM if available.

Each model should be wrapped in try/except so a failure does not crash the notebook.

===============================================================================
PHASE 6: MODEL COMPARISON & BEST MODEL
===============================================================================

Collect all successful evaluation results.

Build a comparison DataFrame:

Regression: Test R², Test RMSE, Test MAE, MAPE, CV mean/std.

Classification: Train/Test accuracy, CV mean/std.

Sort by best primary metric (Test R² or Test Accuracy).

Clearly print the best model and its key scores.

===============================================================================
PHASE 7: EXPLAINABILITY
===============================================================================

If the best model exposes feature_importances_:

Build a feature importance table (top 20).

Plot a horizontal bar chart of top importances.

If SHAP is available and the model is tree‑based:

Sample up to 100 rows from X_test.

Compute SHAP values in a try/except block.

Show a bar summary plot and a detailed summary plot, if possible.

===============================================================================
STYLE & PRESENTATION GUIDELINES
===============================================================================

Use as many cells as needed; do NOT cram.

For each major step:

One markdown cell explaining intent.

One or a few code cells doing the work.

Use bullet lists in markdown where helpful.

Use one statement per line; avoid ';'.

Use concise, meaningful comments (no noisy banners).

Add a short "Quick Summary" near the top (dataset, target, best model idea, key features).

Generate the notebook content now as markdown plus python code blocks.
'''

            def _generate():
                return self.perplexity_generator.generate_notebook(
                    dataset_ref=dataset_info["ref"],
                    dataset_title=dataset_info["title"],
                    dataset_description=dataset_info["description"],
                    custom_prompt=prompt,
                )

            notebook_content = self._retry_operation(
                _generate, operation_name="Generate notebook with Groq AI"
            )

            if notebook_content:
                logger.info("Notebook generated successfully")
                return notebook_content
            else:
                logger.error("Failed to generate notebook content")
                return None

        except Exception as e:
            logger.error(f"Error generating notebook: {str(e)}")
            return None

    def publish_notebook(self, notebook_content: str, dataset_info: Dict[str, Any]) -> bool:
        """Publish the generated notebook to Kaggle with error handling"""
        try:
            logger.info("Publishing notebook to Kaggle...")

            def _publish():
                return self.publisher.publish_notebook(
                    notebook_content=notebook_content,
                    dataset_ref=dataset_info["ref"],
                    dataset_title=dataset_info["title"],
                )

            result = self._retry_operation(_publish, operation_name="Publish notebook to Kaggle")

            if result:
                logger.info("✅ Notebook published successfully!")
                return True
            else:
                logger.error("Failed to publish notebook")
                return False

        except Exception as e:
            logger.error(f"Error publishing notebook: {str(e)}")
            return False

    def run_daily_workflow(self) -> bool:
        """Execute the complete daily notebook generation and publication workflow"""
        try:
            logger.info("=" * 80)
            logger.info("Starting Daily Kaggle Notebook Workflow")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
            logger.info("=" * 80)

            # Step 1: Fetch trending dataset
            dataset_info = self.fetch_trending_dataset()
            if not dataset_info:
                logger.error("Failed to fetch dataset. Aborting workflow.")
                return False

            logger.info(f"Selected Dataset: {dataset_info['title']}")
            logger.info(f"Dataset Reference: {dataset_info['ref']}")

            # Step 2: Generate notebook
            notebook_content = self.generate_notebook(dataset_info)
            if not notebook_content:
                logger.error("Failed to generate notebook. Aborting workflow.")
                return False

            # Step 3: Publish notebook
            success = self.publish_notebook(notebook_content, dataset_info)

            if success:
                logger.info("=" * 80)
                logger.info("✅ Daily workflow completed successfully!")
                logger.info("=" * 80)
                return True
            else:
                logger.error("=" * 80)
                logger.error("❌ Daily workflow failed at publication step")
                logger.error("=" * 80)
                return False

        except Exception as e:
            logger.error(f"Unexpected error in daily workflow: {str(e)}")
            logger.error("=" * 80)
            logger.error("❌ Daily workflow failed with unexpected error")
            logger.error("=" * 80)
            return False


def main():
    """Main entry point for the script"""
    try:
        orchestrator = KaggleNotebookOrchestrator()
        success = orchestrator.run_daily_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
