#!/usr/bin/env python3
"""
Zeus Core Burndown Scheduler
Schedules hourly burndown chart generation using APScheduler
"""
import os
import sys
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from zeus_core.burndown import generate_burndown_chart
from zeus_core.models import create_test_sprint
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hourly_burndown_job(sprint_id: str):
    """Job function to run burndown chart generation"""
    try:
        logger.info(f"Starting hourly burndown job for {sprint_id}")
        generate_burndown_chart(sprint_id)
        logger.info(f"Completed hourly burndown job for {sprint_id}")
    except Exception as e:
        logger.error(f"Error in hourly burndown job: {e}")


def get_current_sprint():
    """Get the current active sprint (mock implementation)"""
    # In a real implementation, this would query your project management system
    # to find the currently active sprint
    return create_test_sprint()


def start_scheduler():
    """Start the APScheduler for hourly burndown generation"""
    scheduler = BlockingScheduler()
    
    # Get current sprint
    current_sprint = get_current_sprint()
    
    # Add hourly job
    scheduler.add_job(
        hourly_burndown_job,
        trigger=IntervalTrigger(hours=1),
        args=[current_sprint.id],
        id='hourly_burndown',
        name='Hourly Burndown Chart Generation',
        replace_existing=True
    )
    
    logger.info(f"Scheduler started for sprint {current_sprint.id}")
    logger.info("Burndown charts will be generated every hour")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


def start_test_scheduler(duration_minutes: int = 2):
    """Start a test scheduler that runs every minute for testing"""
    scheduler = BlockingScheduler()
    
    current_sprint = get_current_sprint()
    
    # Add job that runs every minute for testing
    scheduler.add_job(
        hourly_burndown_job,
        trigger=IntervalTrigger(minutes=1),
        args=[current_sprint.id],
        id='test_burndown',
        name='Test Burndown Chart Generation (Every Minute)',
        replace_existing=True
    )
    
    logger.info(f"Test scheduler started for sprint {current_sprint.id}")
    logger.info(f"Burndown charts will be generated every minute for {duration_minutes} minutes")
    
    # Auto-stop after specified duration
    scheduler.add_job(
        lambda: scheduler.shutdown(),
        trigger='date',
        run_date=datetime.now().replace(second=0, microsecond=0) + 
                 pd.Timedelta(minutes=duration_minutes),
        id='auto_stop'
    )
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Test scheduler stopped by user")
        scheduler.shutdown()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        start_test_scheduler()
    else:
        start_scheduler()
