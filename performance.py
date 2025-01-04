# performance.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, median, stdev
import logging
import json
import pandas as pd

from metrics import TimeoutMetrics, PromptMetrics, TimeoutStats

@dataclass
class PromptMetrics:
    """
    Records metrics for a single prompt execution attempt.
    
    This class captures detailed performance data for each model call, including
    timing information, success/failure status, and prediction results.
    """
    attempt_number: int
    execution_time_seconds: float
    successful: bool
    timeout_metrics: TimeoutMetrics
    error_message: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    meta_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure proper type conversion for metrics"""
        self.execution_time_seconds = float(self.execution_time_seconds)
        if self.prediction is not None:
            self.prediction = str(self.prediction)
        if self.confidence is not None:
            self.confidence = float(self.confidence)

@dataclass
class TimeoutStats:
    """Statistics related to timeout occurrences"""
    total_timeouts: int
    avg_timeout_duration: float
    max_timeout_duration: float
    total_timeout_duration: float

@dataclass
class PerformanceStats:
    """Comprehensive statistics for an evaluation session"""
    prompt_type: str
    model_name: str
    start_time: datetime
    end_time: datetime
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    timeout_attempts: int
    avg_execution_time: float
    median_execution_time: float
    sd_execution_time: float
    timeout_stats: TimeoutStats
    meta_data_averages: Dict[str, float]
    meta_data_sd: Dict[str, float]
    prediction_accuracy: float = 0.0
    prediction_distribution: Dict[str, int] = None
    actual_distribution: Dict[str, int] = None

class DecisionTracker:
    """Tracks and accumulates prediction statistics"""
    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        self.actual_values = []
        self.predicted_values = []
        
    def record_prediction(self, prediction: str, actual: str):
        """Record a single prediction and its actual value"""
        self.total_predictions += 1
        is_correct = prediction.upper() == actual.upper()
        if is_correct:
            self.correct_predictions += 1
        
        self.actual_values.append(actual)
        self.predicted_values.append(prediction)
    
    def get_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return (self.correct_predictions / self.total_predictions) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive prediction statistics"""
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'actual_distribution': pd.Series(self.actual_values).value_counts().to_dict(),
            'predicted_distribution': pd.Series(self.predicted_values).value_counts().to_dict()
        }

class PerformanceTracker:
    """
    Tracks and analyzes performance metrics during evaluation.
    
    This class maintains running statistics about model performance,
    including execution times, success rates, and prediction accuracy.
    """
    def __init__(self, prompt_type: str, model_name: str):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.metrics: List[PromptMetrics] = []
        self.start_time = datetime.now()
        self.decision_tracker = DecisionTracker()

    def record_attempt(self, metrics: PromptMetrics):
        """Record metrics for a single attempt"""
        self.metrics.append(metrics)
        status = "successful" if metrics.successful else "failed"
        timeout_info = ""
        
        if metrics.timeout_metrics.occurred:
            timeout_info = (
                f" (timeout occurred, {metrics.timeout_metrics.retry_count} retries, "
                f"total timeout duration: {metrics.timeout_metrics.total_timeout_duration:.2f}s)"
            )
        
        logging.debug(
            f"Attempt #{metrics.attempt_number} {status} - "
            f"Execution time: {metrics.execution_time_seconds:.2f}s{timeout_info}"
        )

        if not metrics.successful:
            logging.error(f"Error in attempt #{metrics.attempt_number}: {metrics.error_message}")

    def _calculate_timeout_stats(self) -> TimeoutStats:
        """Calculate timeout-specific statistics"""
        timeout_durations = [
            m.timeout_metrics.total_timeout_duration 
            for m in self.metrics 
            if m.timeout_metrics.occurred
        ]
        
        if not timeout_durations:
            return TimeoutStats(0, 0.0, 0.0, 0.0)
            
        return TimeoutStats(
            total_timeouts=len(timeout_durations),
            avg_timeout_duration=mean(timeout_durations),
            max_timeout_duration=max(timeout_durations),
            total_timeout_duration=sum(timeout_durations)
        )

    def _calculate_meta_data_stats(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate average and standard deviation for numeric metadata fields"""
        meta_data_values: Dict[str, List[float]] = {}
        
        for m in self.metrics:
            if m.meta_data is not None:
                for key, val in m.meta_data.items():
                    if isinstance(val, (int, float)):
                        if key not in meta_data_values:
                            meta_data_values[key] = []
                        # Convert nanoseconds to seconds if needed
                        float_val = float(val)
                        if float_val > 1e7:  # Heuristic for nanosecond values
                            float_val /= 1e9
                        meta_data_values[key].append(float_val)
        
        averages = {}
        std_devs = {}
        
        for key, values in meta_data_values.items():
            if values:
                averages[key] = mean(values)
                std_devs[key] = stdev(values) if len(values) > 1 else 0.0
                
        return averages, std_devs

    def _generate_stats(self) -> PerformanceStats:
        """Generate comprehensive statistics from recorded metrics"""
        execution_times = [m.execution_time_seconds for m in self.metrics]
        
        meta_averages, meta_sds = self._calculate_meta_data_stats()
        decision_stats = self.decision_tracker.get_stats()
        
        return PerformanceStats(
            prompt_type=self.prompt_type,
            model_name=self.model_name,
            start_time=self.start_time,
            end_time=datetime.now(),
            total_attempts=len(self.metrics),
            successful_attempts=sum(1 for m in self.metrics if m.successful),
            failed_attempts=sum(1 for m in self.metrics if not m.successful),
            timeout_attempts=sum(1 for m in self.metrics if m.timeout_metrics.occurred),
            avg_execution_time=mean(execution_times) if execution_times else 0.0,
            median_execution_time=median(execution_times) if execution_times else 0.0,
            sd_execution_time=stdev(execution_times) if len(execution_times) > 1 else 0.0,
            timeout_stats=self._calculate_timeout_stats(),
            meta_data_averages=meta_averages,
            meta_data_sd=meta_sds,
            prediction_accuracy=decision_stats['accuracy'],
            prediction_distribution=decision_stats['predicted_distribution'],
            actual_distribution=decision_stats['actual_distribution']
        )

    def save_metrics(self, execution_time: float):
        """Save comprehensive metrics to files"""
        stats = self._generate_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON metrics
        json_path = f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            stats_dict = asdict(stats)
            json.dump(stats_dict, f, indent=2, default=str)
        
        # Save human-readable report
        self._save_text_report(stats, execution_time, timestamp)

    def _save_text_report(
        self, 
        stats: PerformanceStats,
        execution_time: float,
        timestamp: str
    ):
        """Generate and save human-readable performance report"""
        report_path = f"report_{stats.model_name}_{stats.prompt_type}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Performance Report - {stats.model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Prompt Type: {stats.prompt_type}\n")
            f.write(f"Total Duration: {execution_time:.2f}s\n")
            f.write(f"Start Time: {stats.start_time}\n")
            f.write(f"End Time: {stats.end_time}\n\n")
            
            f.write("Execution Statistics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Attempts: {stats.total_attempts}\n")
            f.write(f"Successful: {stats.successful_attempts}\n")
            f.write(f"Failed: {stats.failed_attempts}\n")
            f.write(f"Timeouts: {stats.timeout_attempts}\n\n")
            
            f.write("Prediction Performance\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {stats.prediction_accuracy:.2f}%\n\n")
            
            f.write("Prediction Distribution:\n")
            for value, count in stats.prediction_distribution.items():
                f.write(f"  {value}: {count}\n")
            
            f.write("\nActual Distribution:\n")
            for value, count in stats.actual_distribution.items():
                f.write(f"  {value}: {count}\n\n")
            
            f.write("Timing Statistics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Execution: {stats.avg_execution_time:.4f}s\n")
            f.write(f"Median Execution: {stats.median_execution_time:.4f}s\n")
            f.write(f"Std Dev Execution: {stats.sd_execution_time:.4f}s\n")
            
            if stats.timeout_stats.total_timeouts > 0:
                f.write("\nTimeout Statistics\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Timeouts: {stats.timeout_stats.total_timeouts}\n")
                f.write(f"Average Duration: {stats.timeout_stats.avg_timeout_duration:.4f}s\n")
                f.write(f"Maximum Duration: {stats.timeout_stats.max_timeout_duration:.4f}s\n")
                f.write(f"Total Duration: {stats.timeout_stats.total_timeout_duration:.4f}s\n")
            
            if stats.meta_data_averages:
                f.write("\nModel Metadata Statistics\n")
                f.write("-" * 20 + "\n")
                f.write(f"{'Metric':<25} {'Average (s)':<12} {'Std Dev (s)':<12}\n")
                f.write("-" * 51 + "\n")
                for key in stats.meta_data_averages:
                    avg = stats.meta_data_averages[key]
                    sd = stats.meta_data_sd.get(key, 0.0)
                    f.write(f"{key:<25} {avg:>11.4f}s {sd:>11.4f}s\n")

def save_aggregate_stats(
    session_results: List[PerformanceStats],
    total_duration: float
):
    """
    Save aggregate statistics across all evaluation sessions.
    
    This function generates both JSON and human-readable reports containing
    comprehensive statistics across all evaluation sessions.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate aggregate metrics
    total_predictions = sum(
        s.successful_attempts for s in session_results
    )
    total_timeouts = sum(
        s.timeout_stats.total_timeouts for s in session_results
    )
    avg_accuracy = mean(
        [s.prediction_accuracy for s in session_results]
    )
    
    # Aggregate by model
    model_stats = {}
    for stat in session_results:
        if stat.model_name not in model_stats:
            model_stats[stat.model_name] = {
                'total_attempts': 0,
                'successful_attempts': 0,
                'accuracy': [],
                'avg_execution_time': []
            }
        
        ms = model_stats[stat.model_name]
        ms['total_attempts'] += stat.total_attempts
        ms['successful_attempts'] += stat.successful_attempts
        ms['accuracy'].append(stat.prediction_accuracy)
        ms['avg_execution_time'].append(stat.avg_execution_time)
    
    # Calculate model averages
    for model, stats in model_stats.items():
        stats['avg_accuracy'] = mean(stats['accuracy'])
        stats['avg_execution_time'] = mean(stats['avg_execution_time'])
    
    # Save JSON format
    aggregate_data = {
        'total_duration': total_duration,
        'total_sessions': len(session_results),
        'total_predictions': total_predictions,
        'total_timeouts': total_timeouts,
        'average_accuracy': avg_accuracy,
        'model_performance': model_stats
    }
    
    json_path = f"aggregate_stats_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(aggregate_data, f, indent=2, default=str)
    
    # Save human-readable report
    report_path = f"aggregate_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("Aggregate Performance Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Duration: {total_duration:.2f}s\n")
        f.write(f"Total Sessions: {len(session_results)}\n")
        f.write(f"Total Predictions: {total_predictions}\n")
        f.write(f"Total Timeouts: {total_timeouts}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.2f}%\n\n")
        
        f.write("Model Performance\n")
        f.write("-" * 20 + "\n")
        for model_name, stats in model_stats.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Total Attempts: {stats['total_attempts']}\n")
            f.write(f"  Successful Attempts: {stats['successful_attempts']}\n")
            f.write(f"  Success Rate: {(stats['successful_attempts'] / stats['total_attempts'] * 100):.2f}%\n")
            f.write(f"  Average Accuracy: {stats['avg_accuracy']:.2f}%\n")
            f.write(f"  Average Execution Time: {stats['avg_execution_time']:.4f}s\n")
        
        # Add prompt type analysis
        prompt_stats = {}
        for stat in session_results:
            if stat.prompt_type not in prompt_stats:
                prompt_stats[stat.prompt_type] = {
                    'accuracies': [],
                    'execution_times': [],
                    'timeout_counts': []
                }
            
            ps = prompt_stats[stat.prompt_type]
            ps['accuracies'].append(stat.prediction_accuracy)
            ps['execution_times'].append(stat.avg_execution_time)
            ps['timeout_counts'].append(stat.timeout_stats.total_timeouts)
        
        f.write("\nPrompt Type Performance\n")
        f.write("-" * 20 + "\n")
        for prompt_type, stats in prompt_stats.items():
            f.write(f"\n{prompt_type}:\n")
            f.write(f"  Average Accuracy: {mean(stats['accuracies']):.2f}%\n")
            f.write(f"  Average Execution Time: {mean(stats['execution_times']):.4f}s\n")
            f.write(f"  Total Timeouts: {sum(stats['timeout_counts'])}\n")