# reporting.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from statistics import mean
from pathlib import Path
import json
from dataclasses import asdict, dataclass
import logging
import asyncio
from collections import defaultdict

from .models import (
    PerformanceStats, TimeoutStats, AggregateApiMetadata, 
    EnhancedPerformanceStats
)

@dataclass
class ModelPerformanceMetrics:
    """Detailed performance metrics for a specific model"""
    success_rate: float
    avg_execution_time: float
    timeout_rate: float
    avg_tokens_per_second: float
    avg_first_token_time: float
    token_efficiency: float  # ratio of useful tokens to total tokens

@dataclass
class PromptPerformanceMetrics:
    """Performance metrics for a specific prompt type"""
    success_rate: float
    avg_execution_time: float
    timeout_rate: float
    avg_completion_length: float
    avg_confidence: float

@dataclass
class AggregateReport:
    """Comprehensive report aggregating metrics across all sessions"""
    total_duration: float
    total_sessions: int
    sessions_by_model: Dict[str, int]
    sessions_by_prompt: Dict[str, int]
    overall_success_rate: float
    avg_execution_time: float
    total_timeouts: int
    avg_timeout_duration: float
    model_performance: Dict[str, ModelPerformanceMetrics]
    prompt_performance: Dict[str, PromptPerformanceMetrics]
    token_usage_trends: Dict[str, List[float]]
    cost_efficiency_metrics: Dict[str, float]

class PerformanceReportGenerator:
    """Generates comprehensive performance reports across evaluation sessions"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_aggregate_report(
        self,
        session_results: List[EnhancedPerformanceStats],
        total_duration: float
    ) -> AggregateReport:
        """
        Analyzes results across all evaluation sessions to generate
        comprehensive performance insights.
        """
        # Initialize counters and aggregators
        model_metrics = defaultdict(list)
        prompt_metrics = defaultdict(list)
        total_successes = 0
        total_attempts = 0
        
        # Process each session's results
        for session in session_results:
            # Update basic counters
            total_successes += session.successful_attempts
            total_attempts += session.total_attempts
            
            # Collect model-specific metrics
            model_metrics[session.model_name].append(self._extract_model_metrics(session))
            
            # Collect prompt-specific metrics
            prompt_metrics[session.prompt_type].append(self._extract_prompt_metrics(session))
        
        # Calculate aggregate statistics
        aggregate_report = AggregateReport(
            total_duration=total_duration,
            total_sessions=len(session_results),
            sessions_by_model=self._count_sessions_by_model(session_results),
            sessions_by_prompt=self._count_sessions_by_prompt(session_results),
            overall_success_rate=self._calculate_success_rate(total_successes, total_attempts),
            avg_execution_time=self._calculate_avg_execution_time(session_results),
            total_timeouts=self._count_total_timeouts(session_results),
            avg_timeout_duration=self._calculate_avg_timeout_duration(session_results),
            model_performance=self._aggregate_model_performance(model_metrics),
            prompt_performance=self._aggregate_prompt_performance(prompt_metrics),
            token_usage_trends=self._analyze_token_trends(session_results),
            cost_efficiency_metrics=self._calculate_cost_efficiency(session_results)
        )
        
        # Save the report
        await self._save_aggregate_report(aggregate_report)
        
        return aggregate_report

    async def generate_model_comparison_report(
        self,
        session_results: List[EnhancedPerformanceStats]
    ) -> str:
        """
        Creates a detailed comparison report between different models,
        analyzing their relative strengths and weaknesses.
        """
        model_metrics = self._aggregate_model_metrics(session_results)
        report_content = self._format_model_comparison(model_metrics)
        
        # Save the comparison report
        report_path = self.output_dir / f"model_comparison_{datetime.now():%Y%m%d_%H%M%S}.txt"
        async with aopen(report_path, 'w') as f:
            await f.write(report_content)
        
        return report_content

    async def generate_trend_analysis(
        self,
        session_results: List[EnhancedPerformanceStats]
    ) -> Dict[str, Any]:
        """
        Analyzes performance trends over time, identifying patterns
        and potential areas for optimization.
        """
        trends = {
            'token_efficiency': self._analyze_token_efficiency_trend(session_results),
            'response_times': self._analyze_response_time_trend(session_results),
            'success_rates': self._analyze_success_rate_trend(session_results),
            'timeout_patterns': self._analyze_timeout_patterns(session_results)
        }
        
        # Save trend analysis
        analysis_path = self.output_dir / f"trend_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
        async with aopen(analysis_path, 'w') as f:
            await f.write(json.dumps(trends, indent=2))
        
        return trends

    def _extract_model_metrics(
        self,
        session: EnhancedPerformanceStats
    ) -> ModelPerformanceMetrics:
        """Extracts performance metrics for a specific model"""
        return ModelPerformanceMetrics(
            success_rate=session.successful_attempts / session.total_attempts * 100,
            avg_execution_time=session.avg_execution_time,
            timeout_rate=session.timeout_attempts / session.total_attempts * 100,
            avg_tokens_per_second=session.api_metrics.avg_tokens_per_second,
            avg_first_token_time=session.api_metrics.avg_first_token_time,
            token_efficiency=self._calculate_token_efficiency(session)
        )

    def _extract_prompt_metrics(
        self,
        session: EnhancedPerformanceStats
    ) -> PromptPerformanceMetrics:
        """Extracts performance metrics for a specific prompt type"""
        return PromptPerformanceMetrics(
            success_rate=session.successful_attempts / session.total_attempts * 100,
            avg_execution_time=session.avg_execution_time,
            timeout_rate=session.timeout_attempts / session.total_attempts * 100,
            avg_completion_length=session.api_metrics.avg_completion_tokens,
            avg_confidence=self._calculate_avg_confidence(session)
        )

    def _calculate_token_efficiency(
        self,
        session: EnhancedPerformanceStats
    ) -> float:
        """
        Calculates the ratio of useful output tokens to total tokens used,
        providing insight into prompt efficiency.
        """
        total_tokens = session.api_metrics.total_tokens
        if total_tokens == 0:
            return 0.0
        
        completion_tokens = session.api_metrics.total_completion_tokens
        return (completion_tokens / total_tokens) * 100

    async def _save_aggregate_report(self, report: AggregateReport):
        """Saves the aggregate report in both JSON and human-readable formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON format
        json_path = self.output_dir / f"aggregate_stats_{timestamp}.json"
        async with aopen(json_path, 'w') as f:
            await f.write(json.dumps(asdict(report), indent=2, default=str))
        
        # Save human-readable report
        report_path = self.output_dir / f"aggregate_report_{timestamp}.txt"
        async with aopen(report_path, 'w') as f:
            await f.write(self._format_aggregate_report(report))

    def _format_aggregate_report(self, report: AggregateReport) -> str:
        """Formats the aggregate report for human readability"""
        return f"""Aggregate Performance Analysis Report
==================================

Overall Statistics
-----------------
Total Duration: {report.total_duration:.2f}s
Total Sessions: {report.total_sessions}
Overall Success Rate: {report.overall_success_rate:.2f}%
Average Execution Time: {report.avg_execution_time:.2f}s
Total Timeouts: {report.total_timeouts}
Average Timeout Duration: {report.avg_timeout_duration:.2f}s

Model Performance Summary
-----------------------
{self._format_model_performance(report.model_performance)}

Prompt Performance Summary
------------------------
{self._format_prompt_performance(report.prompt_performance)}

Token Usage Trends
----------------
{self._format_token_trends(report.token_usage_trends)}

Cost Efficiency Metrics
---------------------
{self._format_cost_metrics(report.cost_efficiency_metrics)}
"""

    def _format_model_performance(
        self,
        model_performance: Dict[str, ModelPerformanceMetrics]
    ) -> str:
        """Formats model performance metrics for the report"""
        formatted_sections = []
        for model, metrics in model_performance.items():
            formatted_sections.append(
                f"\n{model}:\n"
                f"  Success Rate: {metrics.success_rate:.2f}%\n"
                f"  Avg Execution Time: {metrics.avg_execution_time:.2f}s\n"
                f"  Timeout Rate: {metrics.timeout_rate:.2f}%\n"
                f"  Token Generation Speed: {metrics.avg_tokens_per_second:.1f} tokens/s\n"
                f"  Token Efficiency: {metrics.token_efficiency:.1f}%"
            )
        return "\n".join(formatted_sections)

    def _analyze_token_efficiency_trend(
        self,
        session_results: List[EnhancedPerformanceStats]
    ) -> List[float]:
        """Analyzes how token efficiency changes over time"""
        return [
            self._calculate_token_efficiency(session)
            for session in sorted(session_results, key=lambda x: x.start_time)
        ]

    def _analyze_response_time_trend(
        self,
        session_results: List[EnhancedPerformanceStats]
    ) -> List[float]:
        """Analyzes how response times change over time"""
        return [
            session.avg_execution_time
            for session in sorted(session_results, key=lambda x: x.start_time)
        ]

    def _calculate_cost_efficiency(
        self,
        session_results: List[EnhancedPerformanceStats]
    ) -> Dict[str, float]:
        """
        Calculates cost efficiency metrics for each model,
        considering token usage and success rates.
        """
        cost_metrics = {}
        for session in session_results:
            total_tokens = session.api_metrics.total_tokens
            success_rate = session.successful_attempts / session.total_attempts
            
            # Calculate cost efficiency score
            # Higher score means better efficiency
            efficiency_score = success_rate * (session.api_metrics.avg_tokens_per_second / total_tokens)
            
            cost_metrics[session.model_name] = efficiency_score
            
        return cost_metrics