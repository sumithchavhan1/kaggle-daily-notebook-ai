# Quick Start Guide: Using Perplexity Tasks

## Overview

This guide helps you get started with Perplexity Task scheduling for your automated Kaggle notebook generation system.

You have created three key files:
- **PERPLEXITY_TASK_GUIDE.md** - Comprehensive reference documentation
- **perplexity_task_scheduler.py** - APScheduler implementation for local/server deployment
- **monitor_perplexity_tasks.py** - Monitoring dashboard to track execution status

## Quick Start (5 Minutes)

### Option A: Using GitHub Actions (Already Set Up - RECOMMENDED)

Your GitHub Actions workflow is already running at 9 AM IST daily via `daily.yml`. This is the primary scheduler.

**Status:** ✓ Already configured and running

### Option B: Using APScheduler (Local/Server)

Use this if you want to run the scheduler on your local machine or cloud server.

#### Step 1: Install Dependencies

```bash
pip install apscheduler
```

#### Step 2: Run the Scheduler

```bash
python perplexity_task_scheduler.py
```

The scheduler will:
- Start immediately
- Schedule daily execution at 9:00 AM IST
- Log all executions to `perplexity_task_logs.json`
- Create `perplexity_scheduler.log` for detailed logs

#### Step 3: Monitor Execution

In another terminal:

```bash
python monitor_perplexity_tasks.py
```

This displays:
- Last execution status
- Success rate (7 days)
- Execution time statistics
- Failure reasons
- Consecutive successful runs

#### Step 4: Stop the Scheduler

```bash
Ctrl+C
```

### Option C: Using Windows Task Scheduler

See detailed instructions in PERPLEXITY_TASK_GUIDE.md (Method 2).

### Option D: Using Linux Cron

See detailed instructions in PERPLEXITY_TASK_GUIDE.md (Method 3).

## File Usage Reference

### perplexity_task_scheduler.py

**Purpose:** Schedules and executes daily notebook generation

**Usage:**

```bash
# Run with default settings (9 AM IST daily)
python perplexity_task_scheduler.py

# For development (test every 5 minutes)
# Edit the file and uncomment:
# scheduler.setup_test_task(interval_minutes=5)
```

**Key Methods:**

- `setup_daily_task(hour, minute, timezone)` - Schedule daily execution
- `setup_test_task(interval_minutes)` - Schedule test execution
- `execute_notebook_generation()` - Main execution function
- `print_schedule()` - Display scheduled jobs

**Output Files:**

- `perplexity_task_logs.json` - Execution logs (JSON format)
- `perplexity_scheduler.log` - Detailed scheduler logs

### monitor_perplexity_tasks.py

**Purpose:** Monitor and report on task execution

**Usage:**

```bash
# Display formatted status report
python monitor_perplexity_tasks.py

# Show last 14 days statistics
python monitor_perplexity_tasks.py --days 14

# Output as JSON
python monitor_perplexity_tasks.py --json

# Export report to file
python monitor_perplexity_tasks.py --export report.json

# Analyze custom log file
python monitor_perplexity_tasks.py --log-file /path/to/logs.json
```

**Output Example:**

```
======================================================================
               PERPLEXITY TASK MONITORING DASHBOARD
======================================================================

[LAST EXECUTION]
  Time: 2024-01-15T09:00:45.123456
  Status: ✓ SUCCESS
  Execution ID: 20240115_090045

[STATISTICS - LAST 7 DAYS]
  Total Executions: 7
  Success Rate: 100.0%
  Consecutive Successes: 7

[EXECUTION TIME STATISTICS]
  Minimum: 145.32 seconds
  Maximum: 187.45 seconds
  Average: 162.18 seconds
  Successful Runs: 7

[FAILURES]
  No failures recorded!

======================================================================
```

## Deployment Scenarios

### Scenario 1: Development on Windows

```bash
# Terminal 1: Start scheduler
python perplexity_task_scheduler.py

# Terminal 2: Monitor in real-time
python monitor_perplexity_tasks.py
```

### Scenario 2: Production on Cloud Server (AWS EC2, DigitalOcean)

```bash
# Install APScheduler
pip install apscheduler

# Create systemd service (Linux)
sudo nano /etc/systemd/system/perplexity-scheduler.service
```

Add:

```ini
[Unit]
Description=Perplexity Task Scheduler
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/kaggle-daily-notebook-ai
ExecStart=/usr/bin/python3 /home/ec2-user/kaggle-daily-notebook-ai/perplexity_task_scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable perplexity-scheduler
sudo systemctl start perplexity-scheduler

# Check status
sudo systemctl status perplexity-scheduler
```

### Scenario 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt && pip install apscheduler

COPY . .

CMD ["python", "perplexity_task_scheduler.py"]
```

Build and run:

```bash
docker build -t kaggle-notebook-scheduler .
docker run -d \
  -e PERPLEXITY_API_KEY=your-key \
  -e KAGGLE_CONFIG_JSON=your-config \
  -v $(pwd)/logs:/app/logs \
  kaggle-notebook-scheduler
```

## Comparison: Which Option to Use?

| Scenario | Recommended Option |
|----------|-------------------|
| Primary automation | GitHub Actions (already running) |
| Local development | APScheduler |
| Windows production | Windows Task Scheduler |
| Linux/Cloud server | APScheduler or Linux Cron |
| Docker environment | APScheduler in Docker |
| Backup/Redundancy | APScheduler on different server |

## Troubleshooting

### Issue: Scheduler not running

```bash
# Check if main.py exists and is executable
python -c "import main; print('main.py OK')"

# Check if APScheduler is installed
pip list | grep apscheduler
```

### Issue: Tasks not executing at scheduled time

```bash
# Verify timezone
python -c "from datetime import datetime; import pytz; print(datetime.now(pytz.timezone('Asia/Kolkata')))"

# Check logs
cat perplexity_scheduler.log | tail -20
```

### Issue: Monitor shows no data

```bash
# Check if log file exists
ls -la perplexity_task_logs.json

# Verify log file format
head -1 perplexity_task_logs.json | python -m json.tool
```

## Advanced Configuration

### Change Execution Time

Edit `perplexity_task_scheduler.py` and modify:

```python
# Line 232: Change from 9:00 AM to different time
scheduler.setup_daily_task(hour=15, minute=30, timezone='Asia/Kolkata')  # 3:30 PM IST
```

### Add Multiple Schedules

```python
# Add morning and evening runs
scheduler.setup_daily_task(hour=9, minute=0, timezone='Asia/Kolkata')
scheduler.setup_daily_task(hour=18, minute=0, timezone='Asia/Kolkata')
```

### Enable Test Mode

Uncomment in `perplexity_task_scheduler.py`:

```python
# scheduler.setup_test_task(interval_minutes=5)  # Test every 5 minutes
```

## Next Steps

1. ✓ GitHub Actions workflow is your primary scheduler
2. (Optional) Run APScheduler as backup for redundancy
3. Use `monitor_perplexity_tasks.py` to verify execution
4. Set up alerts/notifications for failures
5. Archive logs periodically

## Support & Documentation

- Full guide: See `PERPLEXITY_TASK_GUIDE.md`
- Configuration: See `CONFIG.md`
- Implementation details: See `IMPLEMENTATION.md`
- Setup status: See `SETUP_COMPLETED_STATUS.md`

## Summary

**Your system is fully configured with multiple scheduling options:**

- ✓ GitHub Actions (Primary - Running daily at 9 AM IST)
- ✓ APScheduler implementation (Backup/Local development)
- ✓ Monitoring dashboard (Track execution status)
- ✓ Multiple deployment options (Windows/Linux/Docker/Cloud)

You're ready to go! The automation is already active through GitHub Actions.
