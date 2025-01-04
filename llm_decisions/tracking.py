# tracking.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from statistics import mean, median
import json
from pathlib import Path
import logging
from dataclasses import asdict

from .models import (
    PromptMetrics, TimeoutMetrics, ApiCallMetadata,
    PerformanceStats, TimeoutStats, AggregateApiMetadata,
    Decision
)

class ApiMetricsTracker:
    """
    Tracks and analyzes API performance metrics throughout evaluation sessions.
    This class maintains a history of API calls and provides statistical analysis
    of their performance characteristics.
    """
    def __init__(self):
        self.calls: List[ApiCallMetadata] = []

    def record_call(self, metadata: ApiCallMetadata):
        """
        Records metrics for a single API call, validating the data
        before adding it to the history.
        """
        if metadata.total_duration < 0:
            logging.warning("Invalid duration detected in API metadata")
            return
            
        self.calls.append(metadata)
        self._log_call_metrics(metadata)

    def get_aggregate_metrics(self) -> AggregateApiMetadata:
        """
        Calculates comprehensive statistics across all recorded API calls.
        Returns empty metrics if no calls have been recorded.
        """
        if not self.calls:
            return self._create_empty_metrics()

        # Calculate basic aggregates
        total_calls = len(self.calls)
        total_duration = sum(call.total_duration for call in self.calls)
        total_prompt_tokens = sum(call.prompt_tokens for call in self.calls)
        total_completion_tokens = sum(call.completion_tokens for call in self.calls)
        total_tokens = total_prompt_tokens + total_completion_tokens

        # Calculate averages
        avg_duration = total_duration / total_calls
        avg_prompt_tokens = total_prompt_tokens / total_calls
        avg_completion_tokens = total_completion_tokens / total_calls
        avg_tokens = total_tokens / total_calls
        avg_first_token_time = mean(call.first_token_time for call in self.calls)
        avg_tokens_per_second = mean(call.tokens_per_second for call in self.calls)

        # Calculate detailed token distribution statistics
        token_distribution = self._calculate_token_distribution()

        return AggregateApiMetadata(
            total_calls=total_calls,
            total_duration=total_duration,
            avg_duration=avg_duration,
            total_prompt_tokens=total_prompt_tokens,
            avg_prompt_tokens=avg_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            avg_completion_tokens=avg_completion_tokens,
            total_tokens=total_tokens,
            avg_tokens=avg_tokens,
            avg_first_token_time=avg_first_token_time,
            avg_tokens_per_second=avg_tokens_per_second,
            token_distribution=token_distribution
        )

    def _calculate_token_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates detailed statistics about token usage patterns,
        providing insights into model behavior.
        """
        return {
            'prompt_tokens': {
                'min': min(call.prompt_tokens for call in self.calls),
                'max': max(call.prompt_tokens for call in self.calls),
                'mean': mean(call.prompt_tokens for call in self.calls),
                'median': median(call.prompt_tokens for call in self.calls)
            },
            'completion_tokens': {
                'min': min(call.completion_tokens for call in self.calls),
                'max': max(call.completion_tokens for call in self.calls),
                'mean': mean(call.completion_tokens for call in self.calls),
                'median': median(call.completion_tokens for call in self.calls)
            }
        }

    def _log_call_metrics(self, metadata: ApiCallMetadata):
        """
        Logs detailed metrics for monitoring and debugging purposes.
        """
        logging.debug(
            f"API Call Metrics - Duration: {metadata.total_duration:.2f}s, "
            f"Tokens: {metadata.total_tokens}, "
            f"Speed: {metadata.tokens_per_second:.1f} tokens/s"
        )

    @staticmethod
    def _create_empty_metrics() -> AggregateApiMetadata:
        """
        Creates an empty metrics object with zero values for when no
        calls have been recorded.
        """
        return AggregateApiMetadata(
            total_calls=0,
            total_duration=0.0,
            avg_duration=0.0,
            total_prompt_tokens=0,
            avg_prompt_tokens=0.0,
            total_completion_tokens=0,
            avg_completion_tokens=0.0,
            total_tokens=0,
            avg_tokens=0.0,
            avg_first_token_time=0.0,
            avg_tokens_per_second=0.0,
            token_distribution={
                'prompt_tokens': {'min': 0, 'max': 0, 'mean': 0, 'median': 0},
                'completion_tokens': {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
            }
        )

class EnhancedPerformanceTracker:
    """
    Comprehensive performance tracking system that combines traditional metrics
    with enhanced API monitoring capabilities.
    """
    def __init__(self, prompt_type: str, model_name: str):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.metrics: List[PromptMetrics] = []
        self.start_time = datetime.now()
        self.api_metrics_tracker = ApiMetricsTracker()

    async def record_attempt(self, decision: Optional[Decision],
                           timeout_metrics: TimeoutMetrics,
                           api_metadata: ApiCallMetadata,
                           attempt: int):
        """
        Records comprehensive metrics for a single evaluation attempt,
        including both performance and API metrics.
        """
        execution_time = api_metadata.total_duration
        
        metrics = PromptMetrics(
            attempt_number=attempt,
            execution_time_seconds=execution_time,
            successful=decision is not None,
            timeout_metrics=timeout_metrics,
            prediction=str(decision.prediction) if decision else None,
            confidence=float(decision.confidence) if decision else None
        )
        
        self.metrics.append(metrics)
        self.api_metrics_tracker.record_call(api_metadata)
        
        await self._log_attempt_outcome(metrics)

    async def save_metrics(self, total_execution_time: float):
        """
        Saves detailed performance reports in both JSON and human-readable formats.
        Creates separate files for raw data and analysis.
        """
        stats = self.generate_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed metrics in JSON format
        await self._save_json_metrics(stats, timestamp)
        
        # Save human-readable analysis report
        await self._save_text_report(stats, total_execution_time, timestamp)

    def generate_stats(self) -> PerformanceStats:
        """
        Generates comprehensive performance statistics combining traditional
        metrics with API performance data.
        """
        execution_times = [m.execution_time_seconds for m in self.metrics]
        timeout_attempts = sum(1 for m in self.metrics if m.timeout_metrics.occurred)
        
        return PerformanceStats(
            prompt_type=self.prompt_type,
            model_name=self.model_name,
            start_time=self.start_time,
            end_time=datetime.now(),
            total_attempts=len(self.metrics),
            successful_attempts=sum(1 for m in self.metrics if m.successful),
            failed_attempts=sum(1 for m in self.metrics if not m.successful),
            timeout_attempts=timeout_attempts,
            avg_execution_time=mean(execution_times) if execution_times else 0.0,
            median_execution_time=median(execution_times) if execution_times else 0.0,
            timeout_stats=self._calculate_timeout_stats(),
            api_metrics=self.api_metrics_tracker.get_aggregate_metrics()
        )

    async def _save_json_metrics(self, stats: PerformanceStats, timestamp: str):
        """
        Saves raw metrics data in JSON format for programmatic analysis.
        """
        json_path = Path(f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json")
        
        async with aopen(json_path, 'w') as f:
            await f.write(json.dumps(asdict(stats), indent=2, default=str))

    async def _save_text_report(self, stats: PerformanceStats,
                              execution_time: float, timestamp: str):
        """
        Creates a detailed human-readable performance analysis report.
        """
        report_path = Path(f"report_{self.model_name}_{self.prompt_type}_{timestamp}.txt")
        
        async with aopen(report_path, 'w') as f:
            await f.write(self._generate_report_content(stats, execution_time))

    def _calculate_timeout_stats(self) -> TimeoutStats:
        """
        Analyzes timeout patterns and calculates relevant statistics.
        """
        timeout_durations = [
            m.timeout_metrics.total_timeout_duration 
            for m in self.metrics 
            if m.timeout_metrics.occurred
        ]
        
        return TimeoutStats(
            total_timeouts=len(timeout_durations),
            avg_timeout_duration=mean(timeout_durations) if timeout_durations else 0.0,
            max_timeout_duration=max(timeout_durations) if timeout_durations else 0.0,
            total_timeout_duration=sum(timeout_durations)
        )

    async def _log_attempt_outcome(self, metrics: PromptMetrics):
        """
        Logs the outcome of each evaluation attempt for monitoring.
        """
        status = "successful" if metrics.successful else "failed"
        timeout_info = ""
        
        if metrics.timeout_metrics.occurred:
            timeout_info = (
                f" (timeout occurred, {metrics.timeout_metrics.retry_count} retries, "
                f"total timeout duration: {metrics.timeout_metrics.total_timeout_duration:.2f}s)"
            )
        
        logging.info(
            f"Attempt #{metrics.attempt_number} {status} - "
            f"Execution time: {metrics.execution_time_seconds:.2f}s{timeout_info}"
        )

    def _generate_report_content(self, stats: PerformanceStats,
                               execution_time: float) -> str:
        """
        Generates formatted content for the human-readable performance report.
        """
        return f"""Performance Analysis Report - {self.model_name}
{"="* 60}

Model Configuration
------------------
Prompt Type: {stats.prompt_type}
Total Duration: {execution_time:.2f}s
Start Time: {stats.start_time}
End Time: {stats.end_time}

Execution Statistics
------------------
Total Attempts: {stats.total_attempts}
Successful: {stats.successful_attempts}
Failed: {stats.failed_attempts}
Timeouts: {stats.timeout_attempts}

Performance Metrics
-----------------
Average Execution Time: {stats.avg_execution_time:.2f}s
Median Execution Time: {stats.median_execution_time:.2f}s

API Performance
-------------
Total API Calls: {stats.api_metrics.total_calls}
Average Response Time: {stats.api_metrics.avg_duration:.2f}s
Total Tokens Used: {stats.api_metrics.total_tokens}
Average Tokens per Call: {stats.api_metrics.avg_tokens:.1f}
Generation Speed: {stats.api_metrics.avg_tokens_per_second:.1f} tokens/s
Time to First Token: {stats.api_metrics.avg_first_token_time:.3f}s

Token Usage Analysis
------------------
Prompt Tokens:
  Min: {stats.api_metrics.token_distribution['prompt_tokens']['min']:.1f}
  Max: {stats.api_metrics.token_distribution['prompt_tokens']['max']:.1f}
  Mean: {stats.api_metrics.token_distribution['prompt_tokens']['mean']:.1f}
  Median: {stats.api_metrics.token_distribution['prompt_tokens']['median']:.1f}

Completion Tokens:
  Min: {stats.api_metrics.token_distribution['completion_tokens']['min']:.1f}
  Max: {stats.api_metrics.token_distribution['completion_tokens']['max']:.1f}
  Mean: {stats.api_metrics.token_distribution['completion_tokens']['mean']:.1f}
  Median: {stats.api_metrics.token_distribution['completion_tokens']['median']:.1f}

{"Timeout Analysis" if stats.timeout_stats.total_timeouts > 0 else ""}
{"-" * 16 if stats.timeout_stats.total_timeouts > 0 else ""}
{"Total Timeouts: " + str(stats.timeout_stats.total_timeouts) if stats.timeout_stats.total_timeouts > 0 else ""}
{"Average Duration: " + f"{stats.timeout_stats.avg_timeout_duration:.2f}s" if stats.timeout_stats.total_timeouts > 0 else ""}
{"Maximum Duration: " + f"{stats.timeout_stats.max_timeout_duration:.2f}s" if stats.timeout_stats.total_timeouts > 0 else ""}
{"Total Timeout Duration: " + f"{stats.timeout_stats.total_timeout_duration:.2f}s" if stats.timeout_stats.total_timeouts > 0 else ""}
"""