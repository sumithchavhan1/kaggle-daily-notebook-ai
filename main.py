#!/usr/bin/env python3
"""
Main orchestration script for Kaggle Daily Notebook Generation
Fetches trending datasets, generates notebooks, and publishes them daily at 9 AM IST
"""
import os
import json
import logging
from datetime import datetime
import sys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from perplexity_integration import PerplexityNotebookGenerator
    from publish_to_kaggle import KaggleNotebookPublisher
except ImportError as e:
    print(f"Import error: {str(e)}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kaggle_notebook_gen.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KaggLeNotebookOrchestrator:
    """Main orchestrator for daily notebook generation and publication"""
    
    def __init__(self):
        """Initialize the orchestrator with API credentials"""
        try:
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            logger.info('Kaggle API authenticated successfully')
        except Exception as e:
            logger.error(f'Failed to authenticate Kaggle API: {str(e)}')
            raise
        
        self.perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        if not self.perplexity_key:
            raise ValueError('PERPLEXITY_API_KEY environment variable not set')
        
        try:
            self.perplexity_generator = PerplexityNotebookGenerator(self.perplexity_key)
            self.publisher = KaggleNotebookPublisher(self.kaggle_api)
            logger.info('Orchestrator initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize components: {str(e)}')
            raise
    
    def fetch_trending_dataset(self):
        """Fetch a trending dataset from Kaggle"""
        try:
            logger.info('Fetching trending datasets from Kaggle...')
            
            # Get trending datasets using Kaggle API
            # Using simpler parameters to avoid parsing errors
            datasets = self.kaggle_api.dataset_list(
                sort_by='hottest',
                file_type='csv'
            )
            
            if not datasets:
                logger.warning('No trending datasets found')
                return None
            
            # Select the first trending dataset
            selected_dataset = datasets[0]
            logger.info(f'Selected dataset: {selected_dataset.ref}')
            
            return {
                'ref': selected_dataset.ref,
                'title': selected_dataset.title,
                'description': selected_dataset.subtitle if hasattr(selected_dataset, 'subtitle') else selected_dataset.title,
                'size': selected_dataset.size_bytes if hasattr(selected_dataset, 'size_bytes') else 0,
            }
            
        except Exception as e:
            logger.error(f'Error fetching trending dataset: {str(e)}')
            return None
    
    def generate_notebook(self, dataset_info):
        """Generate a complete Kaggle notebook using Perplexity AI"""
        try:
            logger.info(f'Generating notebook for dataset: {dataset_info["title"]}')
            
            prompt = f"""Create a professional Kaggle notebook for analyzing the '{dataset_info['title']}' dataset.

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
            
            notebook_content = self.perplexity_generator.generate_notebook_content(prompt)
            if not notebook_content:
                logger.error('Perplexity returned empty content')
                return None
            
            logger.info('Notebook content generated successfully')
            return notebook_content
            
        except Exception as e:
            logger.error(f'Error generating notebook: {str(e)}')
            return None
    
    def publish_notebook(self, notebook_content, dataset_info):
        """Publish the generated notebook to Kaggle"""
        try:
            logger.info('Publishing notebook to Kaggle...')
            
            # Create notebook title with dataset name
            dataset_name = dataset_info['title'].replace('_', ' ').title()
            notebook_title = f"ML Analysis: {dataset_name} - {datetime.now().strftime('%Y-%m-%d')}"
            
            publication_url = self.publisher.publish_notebook(
                title=notebook_title,
                content=notebook_content,
                dataset_ref=dataset_info['ref'],
                is_private=False
            )
            
            logger.info(f'Notebook published successfully: {publication_url}')
            return publication_url
            
        except Exception as e:
            logger.error(f'Error publishing notebook: {str(e)}')
            return None
    
    def run(self):
        """Execute the complete workflow"""
        try:
            logger.info('='*60)
            logger.info('Starting Kaggle Daily Notebook Generation Workflow')
            logger.info(f'Execution time: {datetime.now().isoformat()}')
            logger.info('='*60)
            
            # Step 1: Fetch trending dataset
            logger.info('\n[STEP 1] Fetching trending dataset...')
            dataset_info = self.fetch_trending_dataset()
            if not dataset_info:
                logger.error('Failed to fetch trending dataset')
                return False
            logger.info(f'Selected dataset: {dataset_info["ref"]}')
            
            # Step 2: Generate notebook content
            logger.info('\n[STEP 2] Generating notebook content with Perplexity AI...')
            notebook_content = self.generate_notebook(dataset_info)
            if not notebook_content:
                logger.error('Failed to generate notebook content')
                return False
            logger.info(f'Generated {len(notebook_content)} characters of notebook content')
            
            # Step 3: Publish to Kaggle
            logger.info('\n[STEP 3] Publishing notebook to Kaggle...')
            publication_url = self.publish_notebook(notebook_content, dataset_info)
            if not publication_url:
                logger.error('Failed to publish notebook')
                return False
            
            logger.info('\n' + '='*60)
            logger.info('Workflow completed successfully!')
            logger.info(f'Published at: {publication_url}')
            logger.info('='*60)
            return True
            
        except Exception as e:
            logger.error(f'Fatal error in workflow: {str(e)}')
            return False


if __name__ == '__main__':
    try:
        orchestrator = KaggLeNotebookOrchestrator()
        success = orchestrator.run()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f'Fatal initialization error: {str(e)}')
        exit(1)
