# Perplexity Task Integration Guide

## Overview
This guide explains how to create scheduled tasks that integrate Perplexity API with your Kaggle notebook automation system. While Perplexity doesn't have a native task scheduler, you can use Python scheduling libraries or system task schedulers to automate Perplexity API calls.

## Three Methods to Create Perplexity Tasks

### Method 1: Using APScheduler (Recommended for Always-On Systems)

APScheduler allows you to schedule Python tasks to run at specific times. This is ideal for running on a Linux server, AWS Lambda, or cloud VPS.

#### Installation
```bash
pip install apscheduler
```

#### Implementation: `perplexity_task_scheduler.py`
```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import main  # Your existing main.py orchestrator
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerplexityTaskScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        
    def setup_daily_task(self, hour=9, minute=0, timezone='Asia/Kolkata'):
        """
        Schedule the notebook generation task to run at specific time (9 AM IST by default)
        """
        trigger = CronTrigger(
            hour=hour,
            minute=minute,
            timezone=timezone,
            day_of_week='0-6'  # Every day
        )
        
        self.scheduler.add_job(
            func=self.execute_notebook_generation,
            trigger=trigger,
            id='daily_kaggle_notebook',
            name='Daily Kaggle Notebook Generation with Perplexity AI',
            misfire_grace_time=3600,  # 1 hour grace period
            replace_existing=True,
            max_instances=1  # Only one instance at a time
        )
        
        logger.info("Scheduled daily notebook generation at 9:00 AM IST")
        
    def execute_notebook_generation(self):
        """
        Execute the full notebook generation workflow
        """
        try:
            logger.info(f"[{datetime.now()}] Starting notebook generation task...")
            
            # Call your existing main orchestrator
            result = main.orchestrate_notebook_generation()
            
            logger.info(f"Notebook generation completed: {result}")
            
            # Log execution status
            self.log_task_execution(success=True, result=result)
            
        except Exception as e:
            logger.error(f"Task failed: {str(e)}")
            self.log_task_execution(success=False, error=str(e))
            raise
    
    def log_task_execution(self, success, result=None, error=None):
        """
        Log task execution details to a file for monitoring
        """
        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'result': result,
            'error': error
        }
        
        try:
            with open('perplexity_task_logs.json', 'a') as f:
                f.write(json.dumps(execution_log) + '\n')
        except Exception as e:
            logger.error(f"Failed to log execution: {str(e)}")
    
    def start(self):
        """
        Start the scheduler
        """
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("APScheduler started")
    
    def stop(self):
        """
        Stop the scheduler gracefully
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("APScheduler stopped")


if __name__ == '__main__':
    scheduler = PerplexityTaskScheduler()
    scheduler.setup_daily_task(hour=9, minute=0, timezone='Asia/Kolkata')
    scheduler.start()
    
    # Keep the scheduler running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
```

### Method 2: Windows Task Scheduler (For Windows 10/11 Systems)

#### Create a Python Script Wrapper: `run_notebook_task.py`
```python
import subprocess
import sys
import json
from datetime import datetime
import os

def run_daily_notebook_task():
    """
    Wrapper script to be called by Windows Task Scheduler
    """
    try:
        print(f"[{datetime.now()}] Starting Kaggle Notebook Generation Task")
        
        # Execute main.py
        result = subprocess.run(
            [sys.executable, 'main.py'],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Log results
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'returncode': result.returncode,
            'stdout': result.stdout[:500],
            'stderr': result.stderr[:500]
        }
        
        with open('task_execution.log', 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        
        print(f"[{datetime.now()}] Task completed with return code: {result.returncode}")
        return result.returncode
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(run_daily_notebook_task())
```

#### Setup Steps in Windows Task Scheduler:
1. Open Task Scheduler (Press `Win+R`, type `taskschd.msc`)
2. Click "Create Task" in the right sidebar
3. **General Tab:**
   - Name: "Daily Kaggle Notebook with Perplexity"
   - Description: "Automated daily notebook generation using Perplexity AI"
   - Select "Run whether user is logged in or not"
   - Select "Run with highest privileges"

4. **Triggers Tab:**
   - Click "New"
   - Choose "Daily"
   - Set time to "09:00:00" (9 AM)
   - Set timezone to "India Standard Time"
   - Check "Enabled"

5. **Actions Tab:**
   - Action: "Start a program"
   - Program: `C:\Python311\python.exe` (adjust path to your Python)
   - Arguments: `run_notebook_task.py`
   - Start in: `C:\path\to\your\project` (full path to your project)

6. **Conditions Tab:**
   - Only start if on AC power (optional)
   - Wake the computer to run this task (check if needed)

7. **Settings Tab:**
   - Allow task to be run on demand (check)
   - Run task as soon as possible after a scheduled start is missed (check)
   - Stop task if it runs for 2 hours (set timeout)

### Method 3: Linux Cron Job (For Linux Servers/VPS)

#### Create Wrapper Script: `run_notebook_task.sh`
```bash
#!/bin/bash

# Set environment variables
export PERPLEXITY_API_KEY="your-api-key"
export KAGGLE_CONFIG_JSON="your-kaggle-config"

# Navigate to project directory
cd /path/to/kaggle-daily-notebook-ai

# Run the notebook generation
python3 main.py >> /var/log/kaggle_notebook.log 2>&1

# Capture exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "[$(date)] Task failed with exit code $EXIT_STATUS" >> /var/log/kaggle_notebook_errors.log
fi

exit $EXIT_STATUS
```

#### Make the script executable:
```bash
chmod +x /path/to/run_notebook_task.sh
```

#### Add Cron Job:
```bash
crontab -e
```

Add this line (9 AM IST = 3:30 AM UTC):
```
30 3 * * * /path/to/run_notebook_task.sh
```

For IST timezone (UTC+5:30), if your server is in UTC:
```
30 3 * * * /path/to/run_notebook_task.sh
```

## Comparison of Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **APScheduler** | Always running, flexible timing, easy to integrate | Requires always-on server | Cloud servers, VPS, Docker |
| **Windows Task Scheduler** | Native to Windows, easy to setup | Windows only | Windows development/deployment |
| **Linux Cron** | Lightweight, reliable | Linux only, less flexible | Linux servers, cloud VPS |
| **GitHub Actions** (Existing) | No server needed, free, integrated with GitHub | API rate limits, slower | GitHub-integrated workflows |

## Monitoring Perplexity Task Execution

Create a monitoring dashboard: `monitor_tasks.py`
```python
import json
from datetime import datetime, timedelta
import os

class PerplexityTaskMonitor:
    def __init__(self, log_file='perplexity_task_logs.json'):
        self.log_file = log_file
    
    def get_last_execution(self):
        """Get the last task execution status"""
        if not os.path.exists(self.log_file):
            return None
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
        return None
    
    def get_success_rate(self, days=7):
        """Calculate success rate for last N days"""
        if not os.path.exists(self.log_file):
            return 0
        
        threshold = datetime.now() - timedelta(days=days)
        successful = 0
        total = 0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                log_entry = json.loads(line)
                timestamp = datetime.fromisoformat(log_entry['timestamp'])
                
                if timestamp > threshold:
                    total += 1
                    if log_entry.get('success'):
                        successful += 1
        
        return (successful / total * 100) if total > 0 else 0
    
    def print_status(self):
        """Print current monitoring status"""
        last_exec = self.get_last_execution()
        success_rate = self.get_success_rate()
        
        print("=" * 50)
        print("Perplexity Task Monitoring Status")
        print("=" * 50)
        
        if last_exec:
            print(f"Last Execution: {last_exec['timestamp']}")
            print(f"Status: {'✓ Success' if last_exec['success'] else '✗ Failed'}")
            if not last_exec['success']:
                print(f"Error: {last_exec.get('error', 'Unknown')}")
        else:
            print("No execution history found")
        
        print(f"7-Day Success Rate: {success_rate:.1f}%")
        print("=" * 50)

if __name__ == '__main__':
    monitor = PerplexityTaskMonitor()
    monitor.print_status()
```

## Deployment Recommendations

### For Development (Local Testing):
- Use **APScheduler** with a Python script running in the background
- Or use **Windows Task Scheduler** on Windows systems

### For Production (24/7 Availability):
- Use **GitHub Actions** (already configured)
- Add **APScheduler** as backup on a cloud server (AWS EC2, DigitalOcean, etc.)
- Use **Linux Cron** on your server

### For Enterprise:
- Combine GitHub Actions + APScheduler for redundancy
- Add monitoring with health checks
- Use database logging instead of file-based logs

## Troubleshooting Perplexity Tasks

### Issue: Task doesn't execute at scheduled time
**Solution:**
- Check timezone configuration (use 'Asia/Kolkata' for IST)
- Verify system clock is synchronized
- Check system/task scheduler logs
- Ensure Perplexity API is accessible from your execution environment

### Issue: Perplexity API rate limits
**Solution:**
- Implement exponential backoff in perplexity_integration.py
- Cache responses for repeated queries
- Consider batching requests

### Issue: Long execution times
**Solution:**
- Set timeout limits
- Optimize Perplexity prompts for faster responses
- Use async calls if available

## Next Steps

1. Choose your preferred scheduling method based on your infrastructure
2. Test the task locally first before deploying
3. Set up monitoring and alerting
4. Document your task configuration for team reference
5. Create backup/failover task on alternative platform

## Integration with Existing Setup

Your current GitHub Actions workflow will continue to work as the primary scheduler. These Perplexity Task methods provide:
- **Local development capability**
- **Backup scheduling if GitHub Actions fails**
- **Higher execution frequency if needed**
- **Better monitoring and logging**

You can run both simultaneously - GitHub Actions as primary, APScheduler as backup.
