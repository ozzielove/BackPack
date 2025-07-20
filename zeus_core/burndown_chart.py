#!/usr/bin/env python3
"""
Burndown Chart System
Implements automated sprint burndown chart generation and scheduling
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import csv
import json
from pathlib import Path

# Configuration
BURNDOWN_OUT = os.getenv('BURNDOWN_OUT', 'charts')
BURNDOWN_BUCKET = os.getenv('BURNDOWN_BUCKET', 'zeus-charts')

class Sprint:
    """Sprint data structure"""
    def __init__(self, sprint_id, name, start_date, end_date, total_points):
        self.id = sprint_id
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.total_points = total_points

def get_current_sprint():
    """Get current active sprint - mock implementation"""
    # In a real implementation, this would fetch from your project management system
    today = datetime.now()
    sprint_start = today - timedelta(days=7)
    sprint_end = today + timedelta(days=7)
    
    return Sprint(
        sprint_id=f"SPRINT-{today.strftime('%Y%m%d')}",
        name=f"Sprint {today.strftime('%Y-%m-%d')}",
        start_date=sprint_start,
        end_date=sprint_end,
        total_points=100
    )

def snapshot_remaining_points(sprint):
    """
    Step 3: Snapshot remaining points for the sprint
    Writes to <charts>/<id>/remaining.csv
    """
    # Create directory structure
    chart_dir = Path(BURNDOWN_OUT) / sprint.id
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = chart_dir / "remaining.csv"
    
    # Mock remaining points calculation
    # In real implementation, this would query your project management system
    current_time = datetime.now()
    days_elapsed = (current_time - sprint.start_date).days
    total_days = (sprint.end_date - sprint.start_date).days
    
    # Simple linear burndown for demo (in reality this would be actual remaining work)
    if days_elapsed <= 0:
        remaining_points = sprint.total_points
    elif days_elapsed >= total_days:
        remaining_points = 0
    else:
        # Add some realistic variance
        ideal_remaining = sprint.total_points * (1 - days_elapsed / total_days)
        variance = sprint.total_points * 0.1 * (0.5 - (days_elapsed % 3) / 6)  # Some realistic variance
        remaining_points = max(0, ideal_remaining + variance)
    
    # Check if file exists to determine if we should append
    file_exists = csv_path.exists()
    
    # Write/append to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if new file
        if not file_exists:
            writer.writerow(['timestamp', 'remaining_points'])
        
        # Write current snapshot
        writer.writerow([current_time.isoformat(), remaining_points])
    
    print(f"✅ Snapshot saved: {remaining_points} points remaining at {current_time}")
    return csv_path

def render_burndown(df, total_points, sprint):
    """
    Step 4: Render burndown chart
    Output: burndown.png
    """
    plt.figure(figsize=(12, 8))
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot actual burndown
    plt.plot(df['timestamp'], df['remaining_points'], 
             marker='o', linewidth=2, label='Actual', color='#2E86AB')
    
    # Calculate and plot ideal burndown line
    sprint_days = (sprint.end_date - sprint.start_date).days
    ideal_dates = [sprint.start_date + timedelta(days=i) for i in range(sprint_days + 1)]
    ideal_points = [total_points * (1 - i/sprint_days) for i in range(sprint_days + 1)]
    
    plt.plot(ideal_dates, ideal_points, 
             linestyle='--', linewidth=2, label='Ideal', color='#A23B72')
    
    # Formatting
    plt.title(f'Burndown Chart - {sprint.name}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Remaining Story Points', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Set y-axis to start from 0
    plt.ylim(0, total_points * 1.1)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save chart
    chart_dir = Path(BURNDOWN_OUT) / sprint.id
    output_path = chart_dir / "burndown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Burndown chart saved: {output_path}")
    return output_path

def upload_to_s3(png_path, sprint_id):
    """
    Step 5: Upload chart to S3
    """
    try:
        s3_client = boto3.client('s3')
        
        # Upload file
        s3_key = f"burndown-charts/{sprint_id}/burndown.png"
        s3_client.upload_file(str(png_path), BURNDOWN_BUCKET, s3_key)
        
        # Generate public URL (assuming bucket allows public read)
        url = f"https://{BURNDOWN_BUCKET}.s3.amazonaws.com/{s3_key}"
        
        print(f"✅ Chart uploaded to S3: {url}")
        return url
        
    except NoCredentialsError:
        print("❌ AWS credentials not found. Skipping S3 upload.")
        return None
    except ClientError as e:
        print(f"❌ S3 upload failed: {e}")
        return None

def post_to_slack(chart_url, sprint):
    """
    Step 5: Post link to Slack/Teams (mock implementation)
    """
    if chart_url:
        message = f"📊 Burndown chart updated for {sprint.name}: {chart_url}"
    else:
        message = f"📊 Burndown chart updated for {sprint.name} (local file only)"
    
    print(f"🔔 Slack notification: {message}")
    # In real implementation, you would use Slack API or webhook here
    return message

def hourly_burndown_job():
    """
    Step 6: Scheduled job that runs every hour
    """
    print(f"\n🔄 Running hourly burndown job at {datetime.now()}")
    
    try:
        # Get current sprint
        sprint = get_current_sprint()
        
        # Step 3: Snapshot remaining points
        csv_path = snapshot_remaining_points(sprint)
        
        # Step 4: Render chart
        df = pd.read_csv(csv_path)
        png_path = render_burndown(df, sprint.total_points, sprint)
        
        # Step 5: Upload and notify
        chart_url = upload_to_s3(png_path, sprint.id)
        post_to_slack(chart_url, sprint)
        
        print("✅ Hourly burndown job completed successfully")
        
    except Exception as e:
        print(f"❌ Hourly burndown job failed: {e}")

def setup_scheduler():
    """
    Step 6: Set up APScheduler for hourly execution
    """
    scheduler = BlockingScheduler()
    
    # Add job to run every 60 minutes
    scheduler.add_job(
        hourly_burndown_job,
        trigger=IntervalTrigger(minutes=60),
        id='burndown_job',
        name='Hourly Burndown Chart Update'
    )
    
    print("📅 Scheduler configured for hourly burndown updates")
    return scheduler

def run_once():
    """Run the burndown process once (for testing)"""
    print("🚀 Running burndown chart generation once...")
    hourly_burndown_job()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # Run once for testing
        run_once()
    elif len(sys.argv) > 1 and sys.argv[1] == "schedule":
        # Start scheduler
        print("🕐 Starting burndown chart scheduler...")
        scheduler = setup_scheduler()
        try:
            scheduler.start()
        except KeyboardInterrupt:
            print("\n⏹️  Scheduler stopped by user")
            scheduler.shutdown()
    else:
        print("Usage:")
        print("  python burndown_chart.py once      # Run once")
        print("  python burndown_chart.py schedule  # Start scheduler")