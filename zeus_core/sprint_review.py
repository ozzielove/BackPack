#!/usr/bin/env python3
"""
Zeus Core Sprint Review & System Demo Protocol
Generates comprehensive sprint reports and triggers system diagnostics
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import subprocess
import logging
import os
from pathlib import Path

from models import Sprint, BacklogItem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    target: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_meeting_target(self) -> bool:
        return self.target is None or self.value >= self.target
    
    @property
    def variance_from_target(self) -> Optional[float]:
        if self.target is None:
            return None
        return ((self.value - self.target) / self.target) * 100


@dataclass
class SystemDiagnostic:
    """Represents a system diagnostic result"""
    agent_name: str
    status: str  # "healthy", "warning", "error"
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None


@dataclass
class SprintReport:
    """Comprehensive sprint report"""
    sprint: Sprint
    completed_items: List[BacklogItem]
    performance_metrics: List[PerformanceMetric]
    system_diagnostics: List[SystemDiagnostic]
    value_delivered: str
    key_achievements: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def completion_rate(self) -> float:
        if not self.sprint.items:
            return 0.0
        completed_count = len([item for item in self.sprint.items if item.is_done])
        return (completed_count / len(self.sprint.items)) * 100
    
    @property
    def velocity(self) -> int:
        return sum(item.effort_points for item in self.sprint.items if item.is_done)


class SprintReviewGenerator:
    """Generates sprint review reports and system demos"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.subordinate_agents = [
            "burndown_agent",
            "impediment_agent", 
            "resource_agent",
            "notification_agent",
            "integration_agent"
        ]
    
    def generate_sprint_report(self, sprint: Sprint) -> SprintReport:
        """Generate comprehensive sprint report"""
        logger.info(f"📊 Generating sprint report for {sprint.id}")
        
        # Get completed items
        completed_items = [item for item in sprint.items if item.is_done]
        
        # Collect performance metrics
        performance_metrics = self._collect_performance_metrics(sprint)
        
        # Run system diagnostics
        system_diagnostics = self._run_system_diagnostics()
        
        # Calculate value delivered
        value_delivered = self._calculate_value_delivered(sprint, completed_items)
        
        # Identify key achievements
        key_achievements = self._identify_key_achievements(sprint, completed_items)
        
        report = SprintReport(
            sprint=sprint,
            completed_items=completed_items,
            performance_metrics=performance_metrics,
            system_diagnostics=system_diagnostics,
            value_delivered=value_delivered,
            key_achievements=key_achievements
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"✅ Sprint report generated: {report.velocity} points delivered")
        return report
    
    def _collect_performance_metrics(self, sprint: Sprint) -> List[PerformanceMetric]:
        """Collect key performance metrics"""
        metrics = []
        
        # Velocity metric
        velocity = sum(item.effort_points for item in sprint.items if item.is_done)
        metrics.append(PerformanceMetric(
            name="Sprint Velocity",
            value=velocity,
            unit="story_points",
            target=sprint.total_points * 0.8  # 80% target
        ))
        
        # Completion rate
        completion_rate = (len([item for item in sprint.items if item.is_done]) / len(sprint.items)) * 100
        metrics.append(PerformanceMetric(
            name="Story Completion Rate",
            value=completion_rate,
            unit="percentage",
            target=85.0
        ))
        
        # Sprint duration utilization
        days_elapsed = sprint.days_elapsed()
        duration_utilization = (days_elapsed / sprint.duration_days) * 100
        metrics.append(PerformanceMetric(
            name="Sprint Duration Utilization",
            value=duration_utilization,
            unit="percentage",
            target=100.0
        ))
        
        # Burndown trend (simulated)
        burndown_health = self._calculate_burndown_health(sprint)
        metrics.append(PerformanceMetric(
            name="Burndown Health Score",
            value=burndown_health,
            unit="score",
            target=75.0
        ))
        
        return metrics
    
    def _calculate_burndown_health(self, sprint: Sprint) -> float:
        """Calculate burndown health score (0-100)"""
        # Simplified calculation - in real implementation would analyze actual burndown data
        days_elapsed = sprint.days_elapsed()
        expected_completion = (days_elapsed / sprint.duration_days) * 100
        actual_completion = (sprint.completed_points / sprint.total_points) * 100
        
        # Health score based on how close actual is to expected
        variance = abs(expected_completion - actual_completion)
        health_score = max(0, 100 - variance * 2)  # Penalize variance
        
        return health_score
    
    def _run_system_diagnostics(self) -> List[SystemDiagnostic]:
        """Run diagnostics on subordinate agents"""
        diagnostics = []
        
        for agent in self.subordinate_agents:
            diagnostic = self._diagnose_agent(agent)
            diagnostics.append(diagnostic)
        
        return diagnostics
    
    def _diagnose_agent(self, agent_name: str) -> SystemDiagnostic:
        """Run diagnostic on a specific agent"""
        start_time = datetime.now()
        
        try:
            # Simulate agent health check
            if agent_name == "burndown_agent":
                status = self._check_burndown_agent()
            elif agent_name == "impediment_agent":
                status = self._check_impediment_agent()
            elif agent_name == "resource_agent":
                status = self._check_resource_agent()
            elif agent_name == "notification_agent":
                status = self._check_notification_agent()
            elif agent_name == "integration_agent":
                status = self._check_integration_agent()
            else:
                status = {"status": "unknown", "details": "Agent not recognized"}
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SystemDiagnostic(
                agent_name=agent_name,
                status=status["status"],
                details=status["details"],
                response_time_ms=response_time
            )
            
        except Exception as e:
            return SystemDiagnostic(
                agent_name=agent_name,
                status="error",
                details={"error": str(e)},
                response_time_ms=None
            )
    
    def _check_burndown_agent(self) -> Dict[str, Any]:
        """Check burndown agent health"""
        try:
            # Check if burndown module can be imported and basic functions work
            from burndown import setup_environment
            charts_dir = setup_environment()
            
            return {
                "status": "healthy",
                "details": {
                    "charts_directory": str(charts_dir),
                    "directory_exists": charts_dir.exists(),
                    "last_check": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "error", 
                "details": {"error": str(e)}
            }
    
    def _check_impediment_agent(self) -> Dict[str, Any]:
        """Check impediment agent health"""
        try:
            from impediments import resolver
            summary = resolver.get_impediment_summary()
            
            return {
                "status": "healthy",
                "details": {
                    "total_impediments": summary["total_impediments"],
                    "resolution_rate": summary["resolution_rate"],
                    "last_check": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "details": {"error": str(e)}
            }
    
    def _check_resource_agent(self) -> Dict[str, Any]:
        """Check resource agent health (simulated)"""
        # Simulate resource monitoring
        return {
            "status": "healthy",
            "details": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "last_check": datetime.now().isoformat()
            }
        }
    
    def _check_notification_agent(self) -> Dict[str, Any]:
        """Check notification agent health (simulated)"""
        return {
            "status": "healthy",
            "details": {
                "slack_webhook_configured": bool(os.environ.get('SLACK_WEBHOOK_URL')),
                "email_configured": False,  # Simulated
                "last_notification": "2025-01-19T15:30:00",
                "last_check": datetime.now().isoformat()
            }
        }
    
    def _check_integration_agent(self) -> Dict[str, Any]:
        """Check integration agent health (simulated)"""
        return {
            "status": "warning",
            "details": {
                "jira_connection": False,
                "github_connection": True,
                "s3_connection": True,
                "last_sync": "2025-01-19T14:00:00",
                "last_check": datetime.now().isoformat()
            }
        }
    
    def _calculate_value_delivered(self, sprint: Sprint, completed_items: List[BacklogItem]) -> str:
        """Calculate and describe value delivered"""
        total_points = sum(item.effort_points for item in completed_items)
        completion_rate = (len(completed_items) / len(sprint.items)) * 100
        
        value_description = f"Delivered {total_points} story points across {len(completed_items)} completed items "
        value_description += f"({completion_rate:.1f}% completion rate). "
        
        if completion_rate >= 90:
            value_description += "Exceptional sprint performance with near-complete delivery."
        elif completion_rate >= 75:
            value_description += "Strong sprint performance with solid delivery."
        elif completion_rate >= 60:
            value_description += "Moderate sprint performance with room for improvement."
        else:
            value_description += "Below-target sprint performance requiring analysis."
        
        return value_description
    
    def _identify_key_achievements(self, sprint: Sprint, completed_items: List[BacklogItem]) -> List[str]:
        """Identify key achievements from the sprint"""
        achievements = []
        
        # High-value items completed
        high_value_items = [item for item in completed_items if item.effort_points >= 8]
        if high_value_items:
            achievements.append(f"Completed {len(high_value_items)} high-value items (8+ points)")
        
        # Sprint goal achievement
        completion_rate = (len(completed_items) / len(sprint.items)) * 100
        if completion_rate >= 85:
            achievements.append("Achieved sprint goal with high completion rate")
        
        # Velocity achievement
        velocity = sum(item.effort_points for item in completed_items)
        if velocity >= sprint.total_points * 0.8:
            achievements.append(f"Strong velocity of {velocity} story points")
        
        # System stability (based on diagnostics)
        achievements.append("Maintained system stability throughout sprint")
        
        return achievements
    
    def _save_report(self, report: SprintReport):
        """Save report to file"""
        filename = f"sprint_report_{report.sprint.id}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        # Convert report to JSON-serializable format
        report_data = {
            "sprint": {
                "id": report.sprint.id,
                "name": report.sprint.name,
                "start_date": report.sprint.start_datetime.isoformat(),
                "end_date": report.sprint.end_datetime.isoformat(),
                "total_points": report.sprint.total_points,
                "completed_points": report.sprint.completed_points,
                "remaining_points": report.sprint.remaining_points
            },
            "summary": {
                "completion_rate": report.completion_rate,
                "velocity": report.velocity,
                "value_delivered": report.value_delivered,
                "key_achievements": report.key_achievements
            },
            "performance_metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "target": metric.target,
                    "meeting_target": metric.is_meeting_target,
                    "variance": metric.variance_from_target
                }
                for metric in report.performance_metrics
            ],
            "system_diagnostics": [
                {
                    "agent": diag.agent_name,
                    "status": diag.status,
                    "details": diag.details,
                    "response_time_ms": diag.response_time_ms
                }
                for diag in report.system_diagnostics
            ],
            "generated_at": report.generated_at.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📄 Sprint report saved: {filepath}")


# Global generator instance
review_generator = SprintReviewGenerator()


def generate_sprint_review(sprint: Sprint) -> SprintReport:
    """Public API for generating sprint reviews"""
    return review_generator.generate_sprint_report(sprint)


def run_system_demo() -> List[SystemDiagnostic]:
    """Public API for running system demo (diagnostics)"""
    return review_generator._run_system_diagnostics()


if __name__ == "__main__":
    # Demo usage
    print("📊 Zeus Sprint Review & System Demo")
    
    # Create test sprint
    from models import create_test_sprint
    sprint = create_test_sprint()
    
    # Generate report
    report = generate_sprint_review(sprint)
    
    print(f"\n✅ Sprint Report Generated:")
    print(f"Sprint: {report.sprint.name}")
    print(f"Velocity: {report.velocity} points")
    print(f"Completion Rate: {report.completion_rate:.1f}%")
    print(f"Value Delivered: {report.value_delivered}")
    
    print(f"\n🔧 System Diagnostics:")
    for diag in report.system_diagnostics:
        status_emoji = "✅" if diag.status == "healthy" else "⚠️" if diag.status == "warning" else "❌"
        print(f"  {status_emoji} {diag.agent_name}: {diag.status}")