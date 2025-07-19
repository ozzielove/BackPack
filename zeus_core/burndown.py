#!/usr/bin/env python3
"""
Zeus Core Burndown Chart Generator
Usage: python -m zeus_core.burndown SPRINT-001
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import json
import requests
from zeus_core.models import create_test_sprint, Sprint


CHART_DIR = Path(os.getenv("BURNDOWN_OUT", "charts")).expanduser().resolve()


def setup_environment():
    """Setup environment variables and directories"""
    os.environ.setdefault('BURNDOWN_OUT', 'charts')
    os.environ.setdefault('BURNDOWN_BUCKET', 'zeus-charts')
    os.environ.setdefault('SKIP_S3', '0')
    
    CHART_DIR.mkdir(exist_ok=True)
    return CHART_DIR


def snapshot_remaining_points(sprint: Sprint):
    """Snapshot remaining points and write to CSV"""
    charts_dir = setup_environment()
    sprint_dir = charts_dir / sprint.id
    sprint_dir.mkdir(exist_ok=True)
    
    csv_path = sprint_dir / 'remaining.csv'
    
    # Create or append to CSV
    current_time = datetime.now()
    remaining = sprint.remaining_points
    
    # Check if CSV exists and read existing data
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = pd.DataFrame(columns=['timestamp', 'remaining_points'])
    
    # Add new data point
    new_row = pd.DataFrame({
        'timestamp': [current_time],
        'remaining_points': [remaining]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"📊 Snapshot saved: {remaining} points remaining at {current_time.strftime('%Y-%m-%d %H:%M')}")
    
    return df, sprint.total_points


def render_burndown(df: pd.DataFrame, total_points: int, sprint: Sprint) -> Path:
    """Render burndown chart and save as PNG"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot actual burndown
    ax.plot(df['timestamp'], df['remaining_points'], 
            marker='o', linewidth=2, markersize=6, 
            color='#e74c3c', label='Actual Burndown')
    
    # Plot ideal burndown line
    start_time = sprint.start_datetime
    end_time = sprint.end_datetime
    ideal_times = [start_time, end_time]
    ideal_points = [total_points, 0]
    
    ax.plot(ideal_times, ideal_points, 
            linestyle='--', linewidth=2, 
            color='#2ecc71', label='Ideal Burndown')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Story Points Remaining', fontsize=12)
    ax.set_title(f'Burndown Chart - {sprint.name}\n{sprint.id}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add current status text
    current_remaining = df['remaining_points'].iloc[-1]
    days_elapsed = sprint.days_elapsed()
    status_text = f'Current: {current_remaining} points remaining\nDay {days_elapsed} of {sprint.duration_days}'
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save chart
    charts_dir = setup_environment()
    output_path = charts_dir / sprint.id / 'burndown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Chart saved: {output_path}")
    return output_path


# Core logic section

# Removed duplicate _runner function - keeping the most complete version below


def upload_to_s3(png_path: Path, sprint_id: str) -> str:
    """Upload chart to S3 and return public URL"""
    if os.environ.get('SKIP_S3', '0') == '1':
        print("⏭️  S3 upload skipped (SKIP_S3=1)")
        return f"file://{png_path.absolute()}"
    
    try:
        s3_client = boto3.client('s3')
        bucket = os.environ['BURNDOWN_BUCKET']
        key = f"burndown/{sprint_id}/burndown-{datetime.now().strftime('%Y%m%d-%H%M')}.png"
        
        s3_client.upload_file(str(png_path), bucket, key)
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        print(f"☁️  Uploaded to S3: {url}")
        return url
        
    except NoCredentialsError:
        print("❌ AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or use SKIP_S3=1")
        return f"file://{png_path.absolute()}"
    except ClientError as e:
        print(f"❌ S3 upload failed: {e}")
        return f"file://{png_path.absolute()}"


def post_to_slack(chart_url: str, sprint: Sprint):
    """Post burndown chart to Slack"""
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("⏭️  Slack notification skipped (no SLACK_WEBHOOK_URL)")
        return
    
    message = {
        "text": f"📊 Burndown Chart Update - {sprint.name}",
        "attachments": [{
            "color": "good",
            "fields": [
                {"title": "Sprint", "value": sprint.id, "short": True},
                {"title": "Remaining Points", "value": str(sprint.remaining_points), "short": True},
                {"title": "Chart", "value": chart_url, "short": False}
            ]
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=message)
        if response.status_code == 200:
            print("💬 Posted to Slack successfully")
        else:
            print(f"❌ Slack post failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Slack notification error: {e}")


# Removed second duplicate _runner function


def _runner(sprint):
    """Internal runner function for burndown chart generation"""
    # Snapshot current state and get data
    df, total_points = snapshot_remaining_points(sprint)
    
    # Render chart
    png_path = render_burndown(df, total_points, sprint)
    
    # Upload to S3 if not skipped
    if not os.environ.get('SKIP_S3', '0') == '1':
        try:
            upload_to_s3(png_path, sprint.id)
        except Exception as e:
            print(f"S3 upload failed: {e}")
    
    return png_path


# ---------- public wrappers (hot-fix) ----------
def hourly_burndown_job(sprint):
    return _runner(sprint)

def generate_burndown_chart(sprint_id):
    from zeus_core.persistence import load_sprint
    return _runner(load_sprint(sprint_id))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m zeus_core.burndown SPRINT-001")
        sys.exit(1)
    
    sprint_id = sys.argv[1]
    generate_burndown_chart(sprint_id)
