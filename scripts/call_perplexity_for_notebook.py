import os
import json
import requests

def call_perplexity_for_notebook(dataset_meta):
    """Call Perplexity API to generate comprehensive notebook JSON with all cells"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    dataset_slug = dataset_meta['dataset_slug']
    dataset_title = dataset_meta['dataset_title']
    
    user_prompt = f"""You are an expert Kaggle data scientist. Generate a COMPLETE, PROFESSIONAL, end-to-end machine learning notebook for:

Dataset: {dataset_title}
Slug: {dataset_slug}
Data Path: /kaggle/input/{dataset_slug.split('/')[1]}/

Create a FULL-FLEDGED notebook with 15-25 cells including:

## REQUIRED STRUCTURE (Generate ALL these cells):

1. **Title & Introduction Markdown Cell**
   - Professional title with emoji
   - Brief dataset overview
   - Key objectives and questions to answer
   - Table of contents

2. **Import Libraries Code Cell**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
   from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
   import xgboost as xgb
   import warnings
   warnings.filterwarnings('ignore')
   %matplotlib inline
   ```

3. **Data Loading Code Cell**
   - Load CSV/data files from /kaggle/input path
   - Display shape

4. **Initial Data Exploration Markdown**
   - Explain what we'll explore

5. **Basic Info Code Cell** - df.head(), df.info(), df.describe()

6. **Missing Values Analysis**
   - Check for nulls
   - Visualization if missing data exists

7. **Data Visualization Section Markdown**

8-12. **Multiple Visualization Code Cells** (5+ different plots):
   - Distribution plots (histograms, kde plots)
   - Count plots for categorical variables
   - Correlation heatmap
   - Box plots for outlier detection
   - Pairplots or scatter plots showing relationships

13. **Feature Engineering Markdown** - Explain preprocessing steps

14. **Feature Engineering Code**
   - Handle missing values if any
   - Encode categorical variables
   - Feature scaling if needed
   - Create new features if applicable

15. **Train-Test Split Code**
   - 80-20 split
   - Separate features and target

16. **Model 1: RandomForest**
   - Training code
   - Predictions

17. **RandomForest Evaluation**
   - Metrics (accuracy/rmse/r2)
   - Confusion matrix or residual plots
   - Feature importance plot

18. **Model 2: XGBoost**
   - Training code
   - Predictions

19. **XGBoost Evaluation**
   - Metrics
   - Comparison with RandomForest
   - Feature importance plot

20. **Results Comparison Markdown**
   - Compare both models
   - Insights and observations

21. **Advanced Insights Code** (Optional)
   - Additional analysis
   - Business recommendations

22. **Conclusions & Next Steps Markdown**
   - Summary of findings
   - Model performance insights
   - Future improvements

## CRITICAL REQUIREMENTS:
- Output ONLY valid JSON (no markdown, no code blocks, no explanations)
- ALL code must be COMPLETE and EXECUTABLE - NO placeholders like '# Your code here'
- Use actual column names from typical datasets (id, name, date, price, category, etc.)
- All imports must be standard Kaggle libraries
- Code must handle both classification and regression scenarios intelligently
- Include proper error handling
- Add professional comments in code
- Use seaborn/matplotlib for beautiful visualizations

## JSON OUTPUT FORMAT:
{{
    "notebook_title": "Professional Title with Context",
    "notebook_slug": "descriptive-analysis-slug-{dataset_slug.split('/')[1]}",
    "description": "Comprehensive ML analysis with EDA, visualization, RandomForest, XGBoost, and insights",
    "cells": [
        {{"type": "markdown", "content": "# Title\n\nDetailed intro..."}},
        {{"type": "code", "content": "import pandas as pd\nimport numpy as np..."}},
        ... (continue with all 15-25 cells)
    ]
}}

Generate the complete JSON NOW with ALL cells filled properly:"""

    try:
        print(f"📡 Calling Perplexity API for comprehensive notebook...")
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
                "max_tokens": 8000
            },
            timeout=90
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
