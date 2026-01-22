# Configuration Guide for Kaggle Daily Notebook AI

## Prerequisites

1. **GitHub Account** - Repository hosting and Actions
2. **Kaggle Account** - For API access and notebook publishing
3. **Perplexity API Key** - For AI-powered notebook generation

## Step 1: Kaggle API Setup

### Get Kaggle API Credentials

1. Log in to [Kaggle.com](https://www.kaggle.com)
2. Navigate to **Settings** → **API**
3. Click **"Create New API Token"**
4. This downloads `kaggle.json` containing your credentials

### Install Kaggle CLI

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Step 2: Perplexity AI API Key

1. Visit [Perplexity AI](https://www.perplexity.ai)
2. Go to your account settings
3. Generate an API key
4. Save this key securely

## Step 3: GitHub Actions Secrets Setup

1. Navigate to your GitHub repository
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Add the following secrets:

### Required Secrets:

**PERPLEXITY_API_KEY**
- Value: Your Perplexity API key from Step 2

**KAGGLE_CONFIG_JSON**
- Value: Contents of your `kaggle.json` file (entire JSON)

**KAGGLE_USERNAME**
- Value: Your Kaggle username

**KAGGLE_KEY**
- Value: Your Kaggle API key from `kaggle.json`

## Step 4: Verify Installation

### Test locally first:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PERPLEXITY_API_KEY="your-key-here"
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"

# Run the main script
python main.py
```

## Step 5: Enable GitHub Actions

1. Go to **Actions** tab in your repository
2. Click **"Enable GitHub Actions"**
3. The workflow will run daily at 9 AM IST (3:30 AM UTC)
4. You can also trigger manually via **Run workflow**

## Schedule Explanation

The workflow is scheduled using cron:
```yaml
- cron: '30 3 * * *'  # 3:30 AM UTC = 9:00 AM IST
```

### Adjusting the Schedule

Edit `.github/workflows/daily.yml`:

- **Daily**: `'0 9 * * *'` (9 AM UTC)
- **Twice daily**: `'0 9,21 * * *'` (9 AM & 9 PM UTC)
- **Weekdays only**: `'0 9 * * 1-5'` (Monday-Friday)

## Troubleshooting

### Issue: "API key error"
- Verify `KAGGLE_CONFIG_JSON` is properly formatted
- Check that username and key match

### Issue: "Perplexity API timeout"
- Check API key is valid
- Verify network connectivity
- Check Perplexity API status

### Issue: "Notebook not publishing"
- Ensure Kaggle API credentials are correct
- Check disk space on GitHub Actions runner
- Review logs in Actions tab

## Monitoring

1. Check **Actions** tab for workflow runs
2. View logs for each execution
3. Published notebooks appear in your [Kaggle Profile](https://www.kaggle.com)

## Security Best Practices

1. Never commit secrets to the repository
2. Rotate API keys regularly
3. Use GitHub's encrypted secrets (automatic)
4. Monitor Actions logs for errors
5. Set up email notifications for failures

## Support & Issues

For issues:
1. Check the Actions logs
2. Verify all secrets are set correctly
3. Test locally before running in Actions
4. Review error messages carefully
