"""
Zeus Core Models - Sprint and BacklogItem classes for burndown tracking
"""
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BacklogItem: 
    effort_points:int
    is_done:bool=False


@dataclass
class Sprint:
    id:str
    name:str
    start_datetime:datetime
    end_datetime:datetime
    items:list[BacklogItem]
    
    @property
    def total_points(self) -> int:
        return sum(item.effort_points for item in self.items)
    
    @property
    def completed_points(self) -> int:
        return sum(item.effort_points for item in self.items if item.is_done)
    
    @property
    def remaining_points(self) -> int:
        return self.total_points - self.completed_points
    
    @property
    def duration_days(self) -> int:
        return (self.end_datetime - self.start_datetime).days
    
    def days_elapsed(self, current_time: Optional[datetime] = None) -> int:
        if current_time is None:
            current_time = datetime.now()
        return max(0, (current_time - self.start_datetime).days)
    
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        if current_time is None:
            current_time = datetime.now()
        return self.start_datetime <= current_time <= self.end_datetime


def create_test_sprint() -> Sprint:
    """Create a dummy sprint for testing purposes"""
    start_date = datetime.now() - timedelta(days=5)
    end_date = datetime.now() + timedelta(days=9)  # 14-day sprint
    
    items = [
        BacklogItem(8, True),   # DONE
        BacklogItem(5, True),   # DONE
        BacklogItem(3, False),  # IN_PROGRESS
        BacklogItem(5, False),  # TODO
        BacklogItem(8, False),  # TODO
        BacklogItem(13, False), # TODO
        BacklogItem(2, False),  # TODO
    ]
    
    return Sprint(
        id='SPRINT-001',
        name='Test Sprint - Q3 Development',
        start_datetime=start_date,
        end_datetime=end_date,
        items=items
    )
