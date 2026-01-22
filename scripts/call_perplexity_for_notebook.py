import os
import json
import requests

def call_perplexity_for_notebook(dataset_meta):
    """Call Perplexity API to generate notebook JSON with all cells"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    dataset_slug = dataset_meta['dataset_slug']
    dataset_title = dataset_meta['dataset_title']
    
    user_prompt = f"""Generate a complete end-to-end Kaggle notebook for:
Dataset Slug: {dataset_slug}
Dataset Title: {dataset_title}
Target: /kaggle/input/{dataset_slug.split('/')[1]}/

Output ONLY valid JSON (no markdown) with this schema:
{{
  "notebook_title": "string",
  "notebook_slug": "string-kebab-case",
  "description": "string",
  "cells": [{{"type": "markdown", "content": "text"}} or {{"type": "code", "content": "Python code"}}]
}}

Include cells for:
1. Problem statement markdown
2. Import libraries (pandas, sklearn, xgboost, matplotlib, seaborn)
3. Load data from /kaggle/input
4. EDA (head, info, describe, missing values, visualizations)
5. Feature engineering
6. Train/test split (80-20)
7. RandomForest model
8. XGBoost model
9. Model evaluation (metrics, confusion matrix, feature importance plots)
10. Conclusions markdown

Code must run end-to-end with NO placeholders."""
    
    try:
        print(f"📡 Calling Perplexity API...")
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "pplx-pro",
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.7,
                "max_tokens": 6000
            },
            timeout=60
        )
        
        response.raise_for_status()
        response_data = response.json()
        notebook_json_str = response_data['choices'][0]['message']['content']
        
        # Parse JSON - handle markdown code blocks
        if "```json" in notebook_json_str:
            notebook_json_str = notebook_json_str.split("```json").split("```")[1]
        elif "```" in notebook_json_str:
            notebook_json_str = notebook_json_str.split("```")[1].split("```")[0]
        
        notebook_spec = json.loads(notebook_json_str.strip())
        print(f"✓ Generated notebook with {len(notebook_spec.get('cells', []))} cells")
        return notebook_spec
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise
