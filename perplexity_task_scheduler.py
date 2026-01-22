#!/usr/bin/env python3
"""
Perplexity Task Scheduler - APScheduler Implementation
Schedules daily Kaggle notebook generation using Perplexity AI
Runs at 9:00 AM IST every day
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import logging
import json
import os
from datetime import datetime
import sys
import signal

# Import your existing main orchestrator
try:
    import main
except ImportError:
    print("Error: main.py not found in current directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perplexity_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerplexityTaskScheduler:
    """
    Scheduler for Perplexity AI-based notebook generation
    """
    
    def __init__(self, log_file='perplexity_task_logs.json'):
        self.scheduler = BackgroundScheduler()
        self.log_file = log_file
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def setup_daily_task(self, hour=9, minute=0, timezone='Asia/Kolkata'):
        """
        Schedule the notebook generation task to run at specific time
        
        Args:
            hour: Hour in 24-hour format (default: 9)
            minute: Minute (default: 0)
            timezone: Timezone string (default: 'Asia/Kolkata' for IST)
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
        
        logger.info(f"Scheduled daily notebook generation at {hour:02d}:{minute:02d} {timezone}")
    
    def setup_test_task(self, interval_minutes=5):
        """
        Setup a test task that runs every N minutes for testing
        
        Args:
            interval_minutes: Interval in minutes (default: 5)
        """
        trigger = IntervalTrigger(minutes=interval_minutes)
        
        self.scheduler.add_job(
            func=self.execute_notebook_generation,
            trigger=trigger,
            id='test_notebook_generation',
            name='Test Notebook Generation (runs every {} minutes)'.format(interval_minutes),
            replace_existing=True,
            max_instances=1
        )
        
        logger.info(f"Scheduled test notebook generation every {interval_minutes} minutes")
    
    def execute_notebook_generation(self):
        """
        Execute the full notebook generation workflow
        """
        execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            logger.info(f"[{execution_id}] Starting notebook generation task...")
            
            # Call your existing main orchestrator
            result = main.orchestrate_notebook_generation()
            
            logger.info(f"[{execution_id}] Notebook generation completed: {result}")
            
            # Log execution status
            self.log_task_execution(
                execution_id=execution_id,
                success=True,
                result=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{execution_id}] Task failed: {str(e)}", exc_info=True)
            self.log_task_execution(
                execution_id=execution_id,
                success=False,
                error=str(e)
            )
            raise
    
    def log_task_execution(self, execution_id, success, result=None, error=None):
        """
        Log task execution details to a JSON file for monitoring
        
        Args:
            execution_id: Unique execution identifier
            success: Whether execution was successful
            result: Result data if successful
            error: Error message if failed
        """
        execution_log = {
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'result': result,
            'error': error
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(execution_log) + '\n')
            logger.debug(f"Logged execution {execution_id}")
        except Exception as e:
            logger.error(f"Failed to log execution {execution_id}: {str(e)}")
    
    def get_scheduled_jobs(self):
        """
        Get list of scheduled jobs
        """
        return self.scheduler.get_jobs()
    
    def print_schedule(self):
        """
        Print current schedule information
        """
        jobs = self.get_scheduled_jobs()
        print("\n" + "="*60)
        print("Perplexity Task Scheduler - Current Schedule")
        print("="*60)
        
        if jobs:
            for job in jobs:
                print(f"\nJob ID: {job.id}")
                print(f"Name: {job.name}")
                print(f"Trigger: {job.trigger}")
                print(f"Next Run: {job.next_run_time}")
        else:
            print("No jobs scheduled")
        
        print("\n" + "="*60 + "\n")
    
    def start(self):
        """
        Start the scheduler
        """
        if not self.running:
            self.scheduler.start()
            self.running = True
            logger.info("APScheduler started successfully")
            self.print_schedule()
        else:
            logger.warning("Scheduler is already running")
    
    def stop(self):
        """
        Stop the scheduler gracefully
        """
        if self.running:
            self.scheduler.shutdown(wait=True)
            self.running = False
            logger.info("APScheduler stopped")
        else:
            logger.warning("Scheduler is not running")
    
    def handle_shutdown(self, signum, frame):
        """
        Handle shutdown signals gracefully
        """
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)


def main_scheduler():
    """
    Main entry point for the scheduler
    """
    logger.info("="*60)
    logger.info("Perplexity Task Scheduler Started")
    logger.info("="*60)
    
    # Create scheduler instance
    scheduler = PerplexityTaskScheduler()
    
    # Setup daily task at 9 AM IST
    scheduler.setup_daily_task(hour=9, minute=0, timezone='Asia/Kolkata')
    
    # Alternatively, for testing, use interval-based scheduling:
    # scheduler.setup_test_task(interval_minutes=5)  # Every 5 minutes
    
    # Start the scheduler
    scheduler.start()
    
    # Keep the scheduler running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        scheduler.stop()


if __name__ == '__main__':
    main_scheduler()
