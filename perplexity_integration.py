#!/usr/bin/env python3
import json
import logging
import time
from typing import Dict
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)

class PerplexityNotebookGenerator:
    """Generate notebook content using Groq AI API (OpenAI compatible)"""
    
    def __init__(self, api_key: str):
        """Initialize with Groq API key"""
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"
        self.timeout = 60
        # Initialize OpenAI client with Groq endpoint
        # Create HTTP client without proxies to avoid compatibility issues
        http_client = httpx.Client()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
                        http_client=http_client
        )
    
    def generate_notebook_content(self, prompt: str) -> str:
        """Generate notebook content from prompt using Groq"""
        try:
            logger.info('Calling Groq API for notebook generation...')
            
            # Combine system and user prompts
            system_prompt = "You are an expert data scientist and machine learning engineer. Generate complete, production-ready Jupyter notebook code with proper markdown documentation."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    logger.info('Successfully generated notebook content with Groq')
                    return self._format_notebook_content(content)
            
            logger.error('Groq API returned empty content')
            return self._generate_template_notebook()
        
        except Exception as e:
            logger.error(f'Groq API error: {str(e)}')
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
        """Generate template notebook as fallback"""
        logger.info('Generating template notebook')
        notebook = {
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Kaggle Notebook: ML Analysis"]}, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import pandas as pd\n", "import numpy as np"]}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        return json.dumps(notebook, indent=2)
