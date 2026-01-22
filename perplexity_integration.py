#!/usr/bin/env python3
import json
import logging
import time
from typing import Dict
import requests

logger = logging.getLogger(__name__)

class PerplexityNotebookGenerator:
    """Generate notebook content using Perplexity AI API"""
    
    def __init__(self, api_key: str):
        """Initialize with Perplexity API key"""
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "pplx-70b-online"
        self.timeout = 60
    
    def generate_notebook_content(self, prompt: str) -> str:
        """Generate notebook content from prompt"""
        try:
            logger.info('Calling Perplexity API for notebook generation...')
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 3000
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content:
                    logger.info('Successfully generated notebook content')
                    return self._format_notebook_content(content)
            
            logger.error(f'API error {response.status_code}')
            return self._generate_template_notebook()
            
        except Exception as e:
            logger.error(f'Error: {str(e)}')
            return self._generate_template_notebook()
    
    def _format_notebook_content(self, content: str) -> str:
        """Format content into notebook structure"""
        notebook = {
            "cells": [],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        sections = content.split('```')
        for i, section in enumerate(sections):
            section = section.strip()
            if not section: continue
            
            if i % 2 == 0:
                notebook['cells'].append({"cell_type": "markdown", "metadata": {}, "source": section.split('\n')})
            else:
                lines = section.split('\n')
                code_start = 1 if lines and lines[0].lower() in ['python', 'python3'] else 0
                notebook['cells'].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": '\n'.join(lines[code_start:]).split('\n')})
        
        return json.dumps(notebook, indent=2)
    
    def _generate_template_notebook(self) -> str:
        """Generate template notebook"""
        logger.info('Generating template notebook')
        notebook = {
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Kaggle Notebook: ML Analysis"]}, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import pandas as pd\n", "import numpy as np"]}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        return json.dumps(notebook, indent=2)
