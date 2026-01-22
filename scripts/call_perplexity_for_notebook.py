import os
import json
import requests

def call_perplexity_for_notebook(dataset_meta):
    """Call Perplexity API to generate comprehensive notebook JSON with all cells"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    dataset_slug = dataset_meta['dataset_slug']
    dataset_title = dataset_meta['dataset_title']
    dataset_path = f"/kaggle/input/{dataset_slug.split('/')[1]}/"
    
    # Simplified prompt without embedded code blocks to avoid API errors
    user_prompt = f"""Generate a complete professional Kaggle notebook analyzing the dataset: {dataset_title}

Dataset path: {dataset_path}

Create a comprehensive ML analysis notebook with 18-22 cells including:

1. Introduction markdown with title and dataset overview
2. Import all necessary libraries: pandas, numpy, matplotlib, seaborn, sklearn, xgboost
3. Load the dataset from the path above
4. Display dataset shape and basic info
5. Show first few rows of data
6. Check for missing values and visualize if present
7. Statistical summary of numeric columns
8. Distribution plots for key numeric features
9. Correlation heatmap
10. Categorical variable analysis with count plots
11. Feature relationships visualization
12. Data preprocessing and feature engineering
13. Split data into train and test sets 80-20
14. Train RandomForest model
15. Evaluate RandomForest with metrics and confusion matrix
16. Train XGBoost model
17. Evaluate XGBoost with metrics and feature importance
18. Compare both models with visualizations
19. Final conclusions and recommendations

IMPORTANT REQUIREMENTS:
- Output ONLY valid JSON, no markdown code blocks
- NO placeholders or comments like TODO or your code here
- All code must be complete and executable
- Use realistic column names like date, price, quantity, category, etc
- Include proper error handling
- Add helpful markdown explanations between code cells

JSON structure:
{{
    "notebook_title": "Professional ML Analysis: {dataset_title}",
    "notebook_slug": "ml-analysis-{dataset_slug.split('/')[1]}",
    "description": "Comprehensive machine learning analysis with EDA, visualizations, RandomForest, and XGBoost models",
    "cells": [
        {{"type": "markdown", "content": "markdown text here"}},
        {{"type": "code", "content": "python code here"}}
    ]
}}

Generate the complete JSON now with all cells filled:"""

    try:
        print(f"📡 Calling Perplexity API for comprehensive notebook...")
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-sonar-large-128k-online",
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.3,
                "max_tokens": 12000
            },
            timeout=120
        )
        
        response.raise_for_status()
        response_data = response.json()
        notebook_json_str = response_data['choices'][0]['message']['content']
        
        # Parse JSON - handle markdown code blocks
        if "```json" in notebook_json_str:
            notebook_json_str = notebook_json_str.split("```json")[1].split("```")[0]
        elif "```" in notebook_json_str:
            notebook_json_str = notebook_json_str.split("```")[1].split("```")[0]
        
        notebook_spec = json.loads(notebook_json_str.strip())
        print(f"✓ Generated notebook with {len(notebook_spec.get('cells', []))} cells")
        return notebook_spec
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise
