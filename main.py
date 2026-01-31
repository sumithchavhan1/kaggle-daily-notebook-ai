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
        """Fetch a trending dataset from Kaggle with retry logic"""
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

                # Choose a random dataset from the top N hottest
                top_n = min(10, len(datasets))
                selected_dataset = random.choice(datasets[:top_n])
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
You are a Kaggle Grandmaster data scientist.

Create a top-quality Kaggle notebook for the dataset '{dataset_info['title']}'.

GOAL:
- Produce a clear, teachable, competition-ready analysis similar in depth and structure to high‑quality Kaggle notebooks.
- Make it easy for readers to learn from the code and text.

STRICT OUTPUT FORMAT:
- Output ONLY a valid Jupyter notebook in JSON (nbformat 4), no backticks, no extra text.
- Use multiple markdown and code cells (not one giant cell).
- Use clean headings: #, ##, ### with short paragraphs and bullet points.
- Code must be PEP8‑style, well‑commented, and logically split into cells.

NOTEBOOK STRUCTURE:

1. INTRO & CONTEXT (markdown)
   - Title with dataset name and date.
   - Brief overview of the problem and what insights/models you will build.
   - Mention key questions you will answer.

2. SETUP & DATA LOADING
   - Code cell: imports (pandas, numpy, matplotlib, seaborn, sklearn, etc.).
   - Code cell: load the main CSV (assume file is in working directory, infer filename if possible).
   - Quick info: shape, dtypes, head, basic .info() and .describe().

3. EXPLORATORY DATA ANALYSIS (EDA)
   - Markdown explaining the EDA plan.
   - Code + plots: missing values, distributions, correlations, time trends if relevant, categorical vs target relationships.
   - Use readable plots with titles, labels, and tight_layout().

4. DATA CLEANING & PREPROCESSING
   - Handle missing values, outliers, type conversions, and encoding of categoricals.
   - Train/test split with clear explanation.
   - Scaling/normalization where appropriate.

5. FEATURE ENGINEERING
   - Create meaningful new features (date parts, ratios, interactions, aggregations) based on dataset context.
   - Explain each important engineered feature in markdown.

6. MODELING
   - Train at least 2–3 models (e.g., Linear/Logistic Regression, Random Forest, Gradient Boosting or XGBoost/LightGBM if installed).
   - Use a consistent evaluation setup (cross‑validation or train/validation split).
   - Show feature importance for tree‑based models when applicable.

7. EVALUATION & COMPARISON
   - Compare models using appropriate metrics (e.g., RMSE/MAE for regression, accuracy/F1/AUC for classification).
   - Present results in a small, readable table.
   - Discuss which model performs best and why.

8. INSIGHTS & CONCLUSIONS
   - Summarize key data insights and model findings.
   - Suggest possible next steps or improvements (e.g., more features, tuning, external data).

GENERAL GUIDELINES:
- Use try/except only where genuinely useful; avoid hiding errors.
- Prefer simple, robust code over overly clever tricks.
- Comment non‑obvious steps.
- Avoid hard‑coding paths other than the main CSV in the current directory.

GENERAL PRESENTATION RULES:
- Always put a blank line after a heading before the paragraph.
- Keep code cells short (20–40 lines each); split long workflows into multiple cells.
- Use one statement per line; avoid `;` to join multiple statements.
- Use clear section headers in markdown that match the Table of Contents.
- Prefer `plt.figure(figsize=(...))` per plot instead of stacking multiple plots in one cell.
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
