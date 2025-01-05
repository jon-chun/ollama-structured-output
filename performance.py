# performance.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, median, stdev
import logging
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from metrics import TimeoutMetrics, PromptMetrics, TimeoutStats


@dataclass
class PerformanceStats:
    """Comprehensive statistics for an evaluation session."""
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
    confusion_matrix: Dict[str, int] = None  # tp, tn, fp, fn
    auc_roc: float = 0.0


class DecisionTracker:
    """Tracks and accumulates prediction statistics across samples."""
    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        self.actual_values = []
        self.predicted_values = []

    def record_prediction(self, prediction: str, actual: str):
        """Record a single prediction & actual."""
        self.total_predictions += 1
        if prediction.upper() == actual.upper():
            self.correct_predictions += 1
        self.actual_values.append(actual)
        self.predicted_values.append(prediction)

    def get_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return 100.0 * self.correct_predictions / self.total_predictions

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a dictionary that includes distributions as well
        as the entire actual/predicted lists for downstream metrics.
        """
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'actual_distribution': pd.Series(self.actual_values).value_counts().to_dict(),
            'predicted_distribution': pd.Series(self.predicted_values).value_counts().to_dict(),
            # Required for confusion matrix/AUC:
            'actual_values': self.actual_values,
            'predicted_values': self.predicted_values
        }


class PerformanceTracker:
    """Maintains metrics about model performance over many attempts."""
    def __init__(self, prompt_type: str, model_name: str):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.metrics: List[PromptMetrics] = []
        self.start_time = datetime.now()
        self.decision_tracker = DecisionTracker()

    def record_attempt(self, metrics: PromptMetrics):
        """Record results from a single attempt."""
        self.metrics.append(metrics)
        status = "successful" if metrics.successful else "failed"

        timeout_info = ""
        if metrics.timeout_metrics.occurred:
            timeout_info = (
                f"(timeout occurred, {metrics.timeout_metrics.retry_count} retries, "
                f"total: {metrics.timeout_metrics.total_timeout_duration:.2f}s)"
            )

        logging.debug(
            f"Attempt #{metrics.attempt_number} {status} - "
            f"execution time: {metrics.execution_time_seconds:.2f}s {timeout_info}"
        )

        # If it was successful and we have a prediction, record it
        if metrics.successful and metrics.prediction is not None and 'actual_value' in (metrics.meta_data or {}):
            actual = metrics.meta_data['actual_value']
            self.decision_tracker.record_prediction(metrics.prediction, actual)

    def _calculate_timeout_stats(self) -> TimeoutStats:
        """Compute aggregated stats about timeouts."""
        timeout_durations = [
            m.timeout_metrics.total_timeout_duration
            for m in self.metrics if m.timeout_metrics.occurred
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
        """
        Averages + std-dev for numeric metadata across all attempts.
        E.g. how long the model took, etc.
        """
        meta_data_values = {}
        for m in self.metrics:
            if m.meta_data is not None:
                # gather numeric keys from meta_data
                for key, val in m.meta_data.items():
                    if isinstance(val, (int, float)):
                        if key not in meta_data_values:
                            meta_data_values[key] = []
                        meta_data_values[key].append(float(val))

        averages = {}
        std_devs = {}
        for key, values in meta_data_values.items():
            if values:
                avg_ = mean(values)
                sd_ = stdev(values) if len(values) > 1 else 0.0
                averages[key] = avg_
                std_devs[key] = sd_

        return averages, std_devs

    def _generate_stats(self) -> PerformanceStats:
        """Compute final PerformanceStats from all recorded attempts."""
        execution_times = [m.execution_time_seconds for m in self.metrics]

        meta_averages, meta_sds = self._calculate_meta_data_stats()
        decision_stats = self.decision_tracker.get_stats()

        # Build y_true / y_pred for confusion matrix and AUC
        y_true = [1 if v.upper() == "YES" else 0 for v in decision_stats['actual_values']]
        y_pred = [1 if v.upper() == "YES" else 0 for v in decision_stats['predicted_values']]

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        confusion_dict = {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

        # AUC-ROC: we treat confidence as probability if available
        # If partial attempts have no confidence, default to 0.5
        y_prob = []
        for m in self.metrics:
            if m.confidence is not None:
                y_prob.append(m.confidence / 100.0)
            else:
                y_prob.append(0.5)

        auc_roc = 0.0
        if len(set(y_true)) > 1:  # avoid error if all are same class
            auc_roc = roc_auc_score(y_true, y_prob)

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
            median_execution_time=median(execution_times) if len(execution_times) > 0 else 0.0,
            sd_execution_time=stdev(execution_times) if len(execution_times) > 1 else 0.0,
            timeout_stats=self._calculate_timeout_stats(),
            meta_data_averages=meta_averages,
            meta_data_sd=meta_sds,
            prediction_accuracy=decision_stats['accuracy'],
            prediction_distribution=decision_stats['predicted_distribution'],
            actual_distribution=decision_stats['actual_distribution'],
            confusion_matrix=confusion_dict,
            auc_roc=auc_roc
        )

    def save_metrics(self, execution_time: float):
        """Save the final stats for this session as JSON and plain text."""
        stats = self._generate_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(stats), f, indent=2, default=str)

        # Text
        self._save_text_report(stats, execution_time, timestamp)

    def _save_text_report(self, stats: PerformanceStats, execution_time: float, timestamp: str):
        """Generate a readable text report of the performance stats."""
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
                f.write(f"  {value}: {count}\n")

            f.write("\nAdditional Classification Metrics\n")
            f.write("-" * 20 + "\n")
            cm = stats.confusion_matrix
            f.write(f"True Positives (TP): {cm['tp']}\n")
            f.write(f"False Positives (FP): {cm['fp']}\n")
            f.write(f"True Negatives (TN): {cm['tn']}\n")
            f.write(f"False Negatives (FN): {cm['fn']}\n")
            f.write(f"AUC-ROC: {stats.auc_roc:.4f}\n\n")

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
                f.write(f"{'Metric':<25} {'Average':<12} {'Std Dev':<12}\n")
                f.write("-" * 51 + "\n")
                for key in stats.meta_data_averages:
                    avg = stats.meta_data_averages[key]
                    sd = stats.meta_data_sd[key]
                    f.write(f"{key:<25} {avg:>11.4f} {sd:>11.4f}\n")


def save_aggregate_stats(session_results: List[PerformanceStats], total_duration: float):
    """
    Summarize multiple sessions (model/prompt combos) into an aggregate report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_predictions = sum(s.successful_attempts for s in session_results)
    total_timeouts = sum(s.timeout_stats.total_timeouts for s in session_results)
    avg_accuracy = mean(s.prediction_accuracy for s in session_results) if session_results else 0.0

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

    # Calculate model-level summaries
    for model, stats in model_stats.items():
        stats['avg_accuracy'] = mean(stats['accuracy']) if stats['accuracy'] else 0.0
        stats['avg_execution_time'] = mean(stats['avg_execution_time']) if stats['avg_execution_time'] else 0.0

    # JSON
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

    # Text summary
    report_path = f"aggregate_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("Aggregate Performance Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("Overall Statistics\n")
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
            success_rate = 0.0
            if stats['total_attempts'] > 0:
                success_rate = (stats['successful_attempts'] / stats['total_attempts'])*100
            f.write(f"  Success Rate: {success_rate:.2f}%\n")
            f.write(f"  Average Accuracy: {stats['avg_accuracy']:.2f}%\n")
            f.write(f"  Average Execution Time: {stats['avg_execution_time']:.4f}s\n")

        # Breakdown by prompt type
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
            if stats['accuracies']:
                f.write(f"  Average Accuracy: {mean(stats['accuracies']):.2f}%\n")
            if stats['execution_times']:
                f.write(f"  Average Execution Time: {mean(stats['execution_times']):.4f}s\n")
            f.write(f"  Total Timeouts: {sum(stats['timeout_counts'])}\n")
