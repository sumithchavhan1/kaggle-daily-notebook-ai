#!/usr/bin/env python3
"""
Perplexity AI Integration for Notebook Generation
Handles communication with Perplexity API to generate notebook content
"""

import json
import logging
import re
from typing import Dict, List
import requests

logger = logging.getLogger(__name__)

class PerplexityNotebookGenerator:
    """Generate notebook content using Perplexity AI API"""
    
    def __init__(self, api_key: str):
        """Initialize with Perplexity API key"""
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "pplx-7b-online"
    
    def generate_notebook_content(self, prompt: str) -> str:
        """Generate notebook content from prompt using Perplexity AI"""
        try:
            logger.info('Calling Perplexity API for notebook generation...')
            
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert data scientist creating comprehensive Kaggle notebooks with ML analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "top_p": 0.9
                }
            )
            
            if response.status_code != 200:
                logger.error(f'Perplexity API error: {response.status_code}')
                raise Exception(f'API request failed: {response.status_code}')
            
            data = response.json()
            notebook_content = data['choices'][0]['message']['content']
            
            logger.info('Successfully generated notebook content')
            return self._format_notebook_content(notebook_content)
        
        except Exception as e:
            logger.error(f'Error in Perplexity API call: {str(e)}')
            raise
    
    def _format_notebook_content(self, content: str) -> Dict:
        """Format the generated content into Jupyter notebook structure"""
        try:
            logger.info('Formatting notebook content...')
            
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            sections = content.split('```')
            
            for i, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue
                
                if i % 2 == 0:
                    notebook['cells'].append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": section.split('\n')
                    })
                else:
                    lines = section.split('\n')
                    code_start = 1 if lines[0] in ['python', 'python3'] else 0
                    code_content = '\n'.join(lines[code_start:])
                    
                    notebook['cells'].append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": code_content.split('\n')
                    })
            
            return json.dumps(notebook, indent=2)
        
        except Exception as e:
            logger.error(f'Error formatting notebook: {str(e)}')
            raise
    
    def generate_markdown_section(self, title: str, content: str) -> str:
        """Generate a markdown cell with title and content"""
        return f"# {title}\n\n{content}"
    
    def generate_code_section(self, code: str, description: str = "") -> str:
        """Generate a code cell with optional description"""
        if description:
            return f"```python\n# {description}\n{code}\n```"
        return f"```python\n{code}\n```"
    
    def generate_ml_section(self, dataset_info: Dict) -> str:
        """Generate ML analysis section template"""
        title = dataset_info.get('title', 'Dataset')
        template = f"""
## Machine Learning Analysis for {title}

This section includes:
1. **Data Preprocessing**: Cleaning and feature engineering
2. **Model Selection**: Multiple ML algorithms
3. **Evaluation**: Cross-validation and metrics
4. **Hyperparameter Tuning**: Optimization
5. **Feature Importance**: Model interpretability
"""
        return template
