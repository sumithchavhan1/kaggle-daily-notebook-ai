# Kaggle API Credentials Setup Guide

This guide explains how to set up KAGGLE_API_TOKEN for the daily Kaggle notebook generation system.

## Step 1: Generate Kaggle API Token

1. Go to https://www.kaggle.com/settings/account
2. Scroll down to the "API" section
3. Click **"Create New API Token"**
   - This will download a file named `kaggle.json`
   - This file contains your API token in JSON format

## Step 2: Extract the API Token

1. Open the downloaded `kaggle.json` file with a text editor
2. The file will contain:
```json
{"username":"your_kaggle_username","key":"your_api_key_here"}
```
3. The entire JSON content is your **KAGGLE_API_TOKEN**

## Step 3: Add GitHub Secret

1. Go to your GitHub repository: https://github.com/sumithchavhan1/kaggle-daily-notebook-ai
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Name: `KAGGLE_API_TOKEN`
5. Value: Paste the entire content of your `kaggle.json` file (the JSON string)
6. Click **"Add secret"**

## Format of KAGGLE_API_TOKEN

The token should be the complete JSON content:
```
{"username":"your_username","key":"your_api_key"}
```

## How It Works

- The GitHub Actions workflow uses this token to authenticate with Kaggle
- During workflow execution:
  1. The token is retrieved from GitHub Secrets
  2. It's written to `~/.kaggle/kaggle.json`
  3. Kaggle CLI tools use this file for authentication
  4. The workflow can now fetch datasets and publish notebooks

## Security Notes

- ✅ Never commit `kaggle.json` to version control
- ✅ Always use GitHub Secrets for sensitive credentials
- ✅ The API token should have read/write permissions for your Kaggle account
- ✅ Rotate your API token if you believe it's been compromised

## Troubleshooting

### Error: "401 - Unauthorized"
This means the API token is invalid or expired.
**Solution**: Generate a new API token and update the GitHub secret.

### Error: "Cannot push notebook"
This could mean the token doesn't have the required permissions.
**Solution**: Ensure you're using a valid, non-expired API token from your Kaggle account.

## Next Steps

Once you've configured the KAGGLE_API_TOKEN secret, the automated workflow will:
1. Fetch new trending datasets from Kaggle daily
2. Generate professional notebooks using Perplexity AI
3. Publish them to your Kaggle profile automatically
