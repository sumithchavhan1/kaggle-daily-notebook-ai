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

            prompt = f"""Create a professional Kaggle notebook for analyzing the '{dataset_info['title']}' dataset.
STRICT FORMAT REQUIREMENTS:
- Output ONLY a Jupyter notebook in JSON matching nbformat 4.
- Use separate markdown and code cells.
- Do NOT wrap JSON in backticks or any prose.
- Use clean markdown headings (##, ###) and short paragraphs.
- Break long code into multiple cells (imports, EDA, preprocessing, modeling, evaluation).
- Use black-compatible, PEP8-style Python code.

Requirements:
1. Import necessary libraries and load the dataset
2. Exploratory Data Analysis (EDA) with visualizations
3. Data cleaning and preprocessing
4. Feature engineering
5. Implement 2-3 machine learning models (e.g., Linear Regression, Random Forest, Gradient Boosting)
6. Model evaluation and comparison with metrics
7. Key insights and recommendations
8. Code should be production-ready with proper error handling
Format as a Jupyter notebook structure with markdown and code cells."""
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

            # Create notebook title with dataset name
            dataset_name = dataset_info["title"].replace("_", " ").title()
            notebook_title = f"ML Analysis: {dataset_name} - {datetime.now().strftime('%Y-%m-%d')}"

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
            logger.info("\n[STEP 1] Fetching trending dataset...")
            dataset_info = self.fetch_trending_dataset()
            if not dataset_info:
                logger.error("Failed to fetch trending dataset")
                return False
            logger.info(f"Selected dataset: {dataset_info['ref']}")

            # Step 2: Generate notebook content
            logger.info("\n[STEP 2] Generating notebook content with Groq AI...")
            notebook_content = self.generate_notebook(dataset_info)
            if not notebook_content:
                logger.error("Failed to generate notebook content")
                return False
            logger.info(f"Generated {len(notebook_content)} characters of notebook content")

            # Step 3: Publish to Kaggle
            logger.info("\n[STEP 3] Publishing notebook to Kaggle...")
            publication_url = self.publish_notebook(notebook_content, dataset_info)
            if not publication_url:
                logger.error("Failed to publish notebook")
                return False

            logger.info("\n" + "=" * 80)
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
