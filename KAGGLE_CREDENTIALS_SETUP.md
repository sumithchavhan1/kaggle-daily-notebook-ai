# Kaggle API Credentials Setup Guide

## Problem: 401 Unauthorized Error

The workflow is failing with `401 - Unauthorized - Unauthorized access` when attempting to push notebooks to Kaggle. This indicates that the Kaggle API credentials are either **invalid** or **expired**.

## Solution: Generate Fresh Kaggle API Credentials

Follow these steps to generate new credentials and update the GitHub secrets:

### Step 1: Generate New Kaggle API Token

1. Go to https://www.kaggle.com/settings/account
2. Scroll down to the **API** section
3. Click **"Create New API Token"**
   - This will download a file named `kaggle.json`
   - The file contains your username and API key

### Step 2: Extract Credentials from kaggle.json

Open the downloaded `kaggle.json` file with a text editor. It will look like:

```json
{"username":"your_username","key":"your_api_key_here"}
```

Note down:
- `username`: Your Kaggle username
- `key`: Your API key

### Step 3: Update GitHub Secrets

1. Go to: https://github.com/sumithchavhan1/kaggle-daily-notebook-ai/settings/secrets/actions
2. Update the following secrets with your fresh credentials:

#### Update `KAGGLE_USERNAME`
- Click the pencil icon next to `KAGGLE_USERNAME`
- Replace the value with your Kaggle username
- Click "Update secret"

#### Update `KAGGLE_KEY`
- Click the pencil icon next to `KAGGLE_KEY`
- Replace the value with your API key from kaggle.json
- Click "Update secret"

#### Delete `KAGGLE_CONFIG_JSON` (Optional)
- Click the trash icon next to `KAGGLE_CONFIG_JSON`
- This secret is no longer needed with the improved workflow

### Step 4: Verify Credentials

1. Go to: https://github.com/sumithchavhan1/kaggle-daily-notebook-ai/actions
2. Click on the workflow "Daily Kaggle Trending Notebook"
3. Click "Run workflow" → "Run workflow"
4. Wait for the workflow to complete
5. Check the logs:
   - Look for: **"Kaggle API authenticated successfully"**
   - If you see authentication errors, verify your credentials are correct

### Step 5: Verify Notebook Publishing

After successful authentication:

1. Wait for the workflow to complete
2. Go to your Kaggle profile: https://www.kaggle.com/sumitchavhan7
3. Click on "Notebooks" tab
4. Look for the newly published notebook with today's date
5. Verify it is **Public** (not Private)

## Troubleshooting

### Error: "401 - Unauthorized"
- **Cause**: Invalid or expired API token
- **Fix**: Regenerate the API token following Step 1-3 above

### Error: "Kernel title does not resolve to specified id"
- **Cause**: Notebook naming issue
- **Fix**: Already addressed in the code, no action needed

### Notebook not appearing on profile
- **Cause**: Privacy settings set to Private
- **Fix**: The workflow should create it as Public, but you can:
  1. Go to the notebook on Kaggle
  2. Click "Share" → Change to "Public"

### Still getting authentication errors after updating secrets
- **Cause**: GitHub Actions may be using cached secrets
- **Fix**: 
  1. Wait 5 minutes
  2. Manually trigger the workflow again
  3. Check the workflow logs in the "Configure Kaggle API" step

## Security Note

⚠️ **Never**:
- Share your API key publicly
- Commit `kaggle.json` to the repository
- Post your API key in issues or comments

Your API key should only be stored in GitHub Secrets, which are encrypted and only accessible during workflow runs.

## Automated Workflow

Once credentials are properly configured:
- The workflow runs automatically at **09:30 AM IST** every day
- It selects a trending dataset from Kaggle
- Generates a Jupyter notebook with ML analysis using Perplexity AI
- Publishes the notebook to your Kaggle profile as **Public**

No further manual intervention is required!
