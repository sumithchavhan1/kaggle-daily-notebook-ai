#!/usr/bin/env python3
"""
Main orchestration script for Kaggle Daily Notebook Generation
Fetches trending datasets, generates notebooks, and publishes them daily at 9 AM IST
"""

import os
import json
import logging
from datetime import datetime
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from perplexity_integration import PerplexityNotebookGenerator
from publish_to_kaggle import KaggleNotebookPublisher

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
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        
        self.perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        if not self.perplexity_key:
            raise ValueError('PERPLEXITY_API_KEY environment variable not set')
        
        self.perplexity_generator = PerplexityNotebookGenerator(self.perplexity_key)
        self.publisher = KaggleNotebookPublisher(self.kaggle_api)
        
        logger.info('Orchestrator initialized successfully')
    
    def fetch_trending_dataset(self):
        """Fetch a trending dataset from Kaggle"""
        try:
            logger.info('Fetching trending datasets from Kaggle...')
            
            # Get trending datasets using Kaggle API
            datasets = self.kaggle_api.dataset_list(
                sort_by='hottest',
                max_size='100MB',
                file_type='csv',
                license_name='cc',
                page=1
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
                'description': selected_dataset.subtitle,
                'size': selected_dataset.size_bytes,
            }
        
        except Exception as e:
            logger.error(f'Error fetching trending dataset: {str(e)}')
            return None
    
    def generate_notebook(self, dataset_info):
        """Generate a complete Kaggle notebook using Perplexity AI"""
        try:
            logger.info(f'Generating notebook for dataset: {dataset_info["title"]}')
            
            prompt = f"""
            Create a comprehensive Kaggle notebook for the dataset: {dataset_info['title']}
            Description: {dataset_info['description']}
            
            Requirements:
            1. Start with data loading and exploration
            2. Include exploratory data analysis (EDA) with visualizations
            3. Perform data cleaning and preprocessing
            4. Implement at least 2 machine learning models
            5. Model evaluation and comparison
            6. Feature engineering techniques
            7. Final recommendations and insights
            8. Code cells should be production-ready
            
            Format the response as a Python Jupyter notebook structure with:
            - Markdown cells for explanations
            - Code cells with implementations
            - Output descriptions
            """
            
            notebook_content = self.perplexity_generator.generate_notebook_content(prompt)
            logger.info('Notebook content generated successfully')
            
            return notebook_content
        
        except Exception as e:
            logger.error(f'Error generating notebook: {str(e)}')
            return None
    
    def publish_notebook(self, notebook_content, dataset_info):
        """Publish the generated notebook to Kaggle"""
        try:
            logger.info('Publishing notebook to Kaggle...')
            
            notebook_title = f"ML Analysis: {dataset_info['title']} - {datetime.now().strftime('%Y-%m-%d')}"
            
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
        logger.info('Starting Kaggle Daily Notebook Generation Workflow...')
        logger.info(f'Execution time: {datetime.now().isoformat()}')
        
        # Step 1: Fetch trending dataset
        dataset_info = self.fetch_trending_dataset()
        if not dataset_info:
            logger.error('Failed to fetch trending dataset')
            return False
        
        # Step 2: Generate notebook content
        notebook_content = self.generate_notebook(dataset_info)
        if not notebook_content:
            logger.error('Failed to generate notebook content')
            return False
        
        # Step 3: Publish to Kaggle
        publication_url = self.publish_notebook(notebook_content, dataset_info)
        if not publication_url:
            logger.error('Failed to publish notebook')
            return False
        
        logger.info('Workflow completed successfully!')
        return True


if __name__ == '__main__':
    try:
        orchestrator = KaggLeNotebookOrchestrator()
        success = orchestrator.run()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f'Fatal error: {str(e)}')
        exit(1)
