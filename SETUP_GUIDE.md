# Kaggle Daily AI Notebook - Complete Setup Guide

## STEP 1: Create Scripts (Copy Below Python Files)

### scripts/__init__.py
# Empty file

### scripts/orchestrate_automation.py
from datetime import datetime
from pathlib import Path
import json
from fetch_trending_dataset import fetch_trending_dataset
from call_perplexity_for_notebook import call_perplexity_for_notebook
from build_notebook import build_notebook_from_spec
from publish_to_kaggle import publish_notebook_to_kaggle

def main():
    print("[STEP 1] Fetching trending dataset...")
    dataset_meta = fetch_trending_dataset()
    
    print("[STEP 2] Calling Perplexity for notebook generation...")
    notebook_spec = call_perplexity_for_notebook(dataset_meta)
    
    print("[STEP 3] Building Jupyter notebook...")
    notebook_info = build_notebook_from_spec(notebook_spec, dataset_meta)
    
    print("[STEP 4] Publishing to Kaggle...")
    publish_notebook_to_kaggle(notebook_info, dataset_meta)
    
    print("✓ Pipeline completed successfully!")

if __name__ == "__main__":
    main()

## STEP 2: GitHub Secrets Setup

Go to: https://github.com/sumithchavhan1/kaggle-daily-notebook-ai/settings/secrets/actions

Add these 3 secrets:

1. KAGGLE_USERNAME
   Value: Your Kaggle username

2. KAGGLE_KEY
   Value: Get from https://www.kaggle.com/settings/account
   Click "Create New Token" - saves kaggle.json
   Copy the 'key' value from the JSON

3. PERPLEXITY_API_KEY
   Value: Get from https://www.perplexity.com/api
   Create API key from dashboard

## STEP 3: Configure Kaggle API

1. Download kaggle.json from https://www.kaggle.com/settings/account
2. Place in ~/.kaggle/kaggle.json
3. Run: chmod 600 ~/.kaggle/kaggle.json

## STEP 4: Perplexity Task Configuration

Go to: https://www.perplexity.com/tasks

Create NEW TASK with these settings:

**Task Name:** Daily Kaggle Notebook Generator
**Schedule:** Daily, 09:00 IST (Asia/Kolkata)
**Model:** pplx-pro or pplx-alpha

**Prompt:**
```
You are a Kaggle Grandmaster. Input will be JSON with dataset info.
Output ONLY valid JSON (no markdown) with this schema:

{
  "notebook_title": "string",
  "notebook_slug": "string-kebab-case",
  "description": "string",
  "cells": [{"type": "markdown"|"code", "content": "string"}]
}

Include cells for:
1. Intro markdown
2. Imports (pandas, sklearn, xgboost, etc)
3. Data loading from /kaggle/input
4. EDA (head, info, describe, missing values)
5. Feature engineering
6. Train/test split (80-20)
7. ML models: RandomForest + XGBoost
8. Model evaluation (metrics + plots)
9. Conclusion markdown

Code must run end-to-end with NO placeholders.
```

## STEP 5: Create Remaining Python Scripts

Your scripts folder should have all 5 files:
- orchestrate_automation.py (main orchestrator)
- fetch_trending_dataset.py (get trending datasets)
- call_perplexity_for_notebook.py (API call)
- build_notebook.py (create .ipynb)
- publish_to_kaggle.py (publish kernel)

## STEP 6: GitHub Actions Verification

Go to: https://github.com/sumithchavhan1/kaggle-daily-notebook-ai/actions

You should see workflow running daily at 9 AM IST (3:30 AM UTC).

## IMPORTANT NOTES

✓ Repository is already created: kaggle-daily-notebook-ai
✓ .github/workflows/daily.yml is created
✓ requirements.txt is created
✓ GitHub Actions scheduled for 9 AM IST daily

Remaining steps:
1. Add 3 GitHub Secrets (KAGGLE_USERNAME, KAGGLE_KEY, PERPLEXITY_API_KEY)
2. Add all 5 Python scripts to scripts/ folder
3. Create Perplexity Task with exact prompt above
4. Create trending_notebooks/ folder with .gitkeep
5. Update README.md with setup instructions
