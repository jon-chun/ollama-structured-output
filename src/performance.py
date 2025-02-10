# performance.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, median, stdev
import logging
import json
import re
import pandas as pd
from pathlib import Path
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
        self.confidences = []

    def record_prediction(self, prediction: str, actual: str, confidence: Optional[float] = None):
        """Record a single prediction & actual."""
        self.total_predictions += 1
        if prediction.upper() == actual.upper():
            self.correct_predictions += 1
        self.actual_values.append(actual)
        self.predicted_values.append(prediction)
        self.confidences.append(confidence if confidence is not None else 0.5)

    def get_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return 100.0 * self.correct_predictions / self.total_predictions

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive prediction statistics."""
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'actual_distribution': pd.Series(self.actual_values).value_counts().to_dict(),
            'predicted_distribution': pd.Series(self.predicted_values).value_counts().to_dict(),
            'actual_values': self.actual_values,
            'predicted_values': self.predicted_values,
            'confidences': self.confidences
        }


class PerformanceTracker:
    """Maintains metrics about model performance over many attempts."""
    def __init__(self, prompt_type: str, model_name: str, output_base_dir: Path):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.output_base_dir = output_base_dir
        self.metrics: List[PromptMetrics] = []
        self.attempts: List[PromptMetrics] = []
        self.start_time = datetime.now()
        self.decision_tracker = DecisionTracker()

    def record_attempt(self, metrics: PromptMetrics):
        """Record results from a single attempt."""
        self.metrics.append(metrics)
        self.attempts.append(metrics)  # For backwards compatibility
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

        # Record prediction if available
        if metrics.successful and metrics.prediction is not None:
            actual_value = metrics.meta_data.get('actual_value') if metrics.meta_data else None
            if actual_value:
                self.decision_tracker.record_prediction(
                    metrics.prediction,
                    actual_value,
                    metrics.confidence
                )

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
        """Calculate averages and std-dev for numeric metadata."""
        meta_data_values = {}
        for m in self.metrics:
            if m.meta_data is not None:
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



    def _generate_stats(self) -> Optional[PerformanceStats]:
        """Compute final PerformanceStats from all recorded attempts."""
        if not self.metrics:
            return None

        execution_times = [m.execution_time_seconds for m in self.metrics]
        meta_averages, meta_sds = self._calculate_meta_data_stats()
        decision_stats = self.decision_tracker.get_stats()

        # Build confusion matrix data only from successful API calls.
        valid_actuals = decision_stats['actual_values']
        valid_preds = decision_stats['predicted_values']

        # Decide which labels to use based on the type of the first value.
        if valid_actuals and isinstance(valid_actuals[0], str):
            cm = confusion_matrix(valid_actuals, valid_preds, labels=["YES", "NO"])
        elif valid_actuals and isinstance(valid_actuals[0], (int, float)):
            cm = confusion_matrix(valid_actuals, valid_preds, labels=[0, 1])
        else:
            import numpy as np
            cm = np.zeros((2, 2), dtype=int)

        # Ensure cm is 2x2 even if one class is missing.
        if cm.shape != (2, 2):
            import numpy as np
            full_cm = np.zeros((2, 2), dtype=int)
            for i in range(min(cm.shape[0], 2)):
                for j in range(min(cm.shape[1], 2)):
                    full_cm[i, j] = cm[i, j]
            cm = full_cm

        tn, fp, fn, tp = cm.ravel()
        confusion_dict = {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

        # Calculate AUC-ROC if possible.
        if valid_actuals and isinstance(valid_actuals[0], str):
            y_true = [1 if v.upper() == "YES" else 0 for v in valid_actuals]
            y_pred = [1 if v.upper() == "YES" else 0 for v in valid_preds]
        else:
            y_true = valid_actuals
            y_pred = valid_preds

        auc_roc = 0.0
        if y_true and len(set(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(y_true, self.decision_tracker.confidences)

        return PerformanceStats(
            prompt_type=str(self.prompt_type),
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
            prediction_accuracy=self.decision_tracker.get_accuracy(),
            prediction_distribution=decision_stats['predicted_distribution'],
            actual_distribution=decision_stats['actual_distribution'],
            confusion_matrix=confusion_dict,
            auc_roc=auc_roc
        )


    def save_metrics(self, execution_time: float):
        """
        Save the final stats for this session as JSON and plain text in the reports directory.
        Creates a separate reports directory for each model to store all statistics.
        """
        stats = self._generate_stats()
        if stats is None:
            logging.warning("No metrics to save - no attempts recorded")
            return

        # Get proper prompt type string for filenames
        prompt_type_mapping = {
            'PromptType.SYSTEM1': 'system1',
            'system1': 'system1',
            'PromptType.COT': 'cot',
            'cot': 'cot',
            'PromptType.COT_NSHOT': 'cot-nshot',
            'cot-nshot': 'cot-nshot'
        }
        prompt_type_str = prompt_type_mapping.get(str(self.prompt_type), str(self.prompt_type))

        # Create model's reports directory
        clean_name = re.sub(r'[^\w\-_]', '_', self.model_name.lower()) + '_reports'
        # reports_dir = self.output_base_dir / clean_name / 'reports'
        reports_dir = self.output_base_dir / clean_name 
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON format in reports directory
        json_path = reports_dir / f"metrics_{self.model_name}_{prompt_type_str}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(stats), f, indent=2, default=str)
            logging.info(f"Saved metrics JSON to {json_path}")

        # Save text report in reports directory
        self._save_text_report(
            stats=stats,
            execution_time=execution_time,
            timestamp=timestamp,
            output_dir=reports_dir,
            prompt_type_str=prompt_type_str
        )
        logging.info(f"Saved performance report to {reports_dir}")

    def _save_text_report(
        self,
        stats: PerformanceStats,
        execution_time: float,
        timestamp: str,
        output_dir: Path,
        prompt_type_str: str
    ):
        """Generate a readable text report of the performance stats."""
        report_path = output_dir / f"report_{self.model_name}_{prompt_type_str}_{timestamp}.txt"
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
            for value, count in (stats.prediction_distribution or {}).items():
                f.write(f"  {value}: {count}\n")

            f.write("\nActual Distribution:\n")
            for value, count in (stats.actual_distribution or {}).items():
                f.write(f"  {value}: {count}\n")

            f.write("\nClassification Metrics\n")
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
    """Summarize multiple sessions into an aggregate report."""
    if not session_results:
        logging.warning("No session results to aggregate")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("metrics")
    output_base.mkdir(parents=True, exist_ok=True)

    total_predictions = sum(s.successful_attempts for s in session_results)
    total_timeouts = sum(s.timeout_stats.total_timeouts for s in session_results)
    avg_accuracy = mean(s.prediction_accuracy for s in session_results)

    # Aggregate by model
    model_stats = {}
    for stat in session_results:
        if stat.model_name not in model_stats:
            model_stats[stat.model_name] = {
                'total_attempts': 0,
                'successful_attempts': 0,
                'accuracy': [],
                'avg_execution_time': [],
                'auc_roc': [],
                'timeouts': 0
            }
        ms = model_stats[stat.model_name]
        ms['total_attempts'] += stat.total_attempts
        ms['successful_attempts'] += stat.successful_attempts
        ms['accuracy'].append(stat.prediction_accuracy)
        ms['avg_execution_time'].append(stat.avg_execution_time)
        ms['auc_roc'].append(stat.auc_roc)
        ms['timeouts'] += stat.timeout_stats.total_timeouts

    # Calculate model-level summaries
    for model, stats in model_stats.items():
        stats['avg_accuracy'] = mean(stats['accuracy']) if stats['accuracy'] else 0.0
        stats['avg_execution_time'] = mean(stats['avg_execution_time']) if stats['avg_execution_time'] else 0.0
        stats['avg_auc_roc'] = mean(stats['auc_roc']) if stats['auc_roc'] else 0.0
        if stats['total_attempts'] > 0:
            stats['success_rate'] = (stats['successful_attempts'] / stats['total_attempts']) * 100
        else:
            stats['success_rate'] = 0.0

    # Save JSON output
    aggregate_data = {
        'timestamp': timestamp,
        'total_duration': total_duration,
        'total_sessions': len(session_results),
        'total_predictions': total_predictions,
        'total_timeouts': total_timeouts,
        'average_accuracy': avg_accuracy,
        'model_performance': model_stats
    }

    json_path = output_base / f"aggregate_stats_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(aggregate_data, f, indent=2, default=str)

    # Generate text report
    report_path = output_base / f"aggregate_report_{timestamp}.txt"
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
            f.write(f"  Success Rate: {stats['success_rate']:.2f}%\n")
            f.write(f"  Average Accuracy: {stats['avg_accuracy']:.2f}%\n")
            f.write(f"  Average AUC-ROC: {stats['avg_auc_roc']:.4f}\n")
            f.write(f"  Average Execution Time: {stats['avg_execution_time']:.4f}s\n")
            f.write(f"  Total Timeouts: {stats['timeouts']}\n")

        # Breakdown by prompt type
        prompt_stats = {}
        for stat in session_results:
            if stat.prompt_type not in prompt_stats:
                prompt_stats[stat.prompt_type] = {
                    'accuracies': [],
                    'execution_times': [],
                    'timeout_counts': [],
                    'auc_rocs': []
                }
            ps = prompt_stats[stat.prompt_type]
            ps['accuracies'].append(stat.prediction_accuracy)
            ps['execution_times'].append(stat.avg_execution_time)
            ps['timeout_counts'].append(stat.timeout_stats.total_timeouts)
            ps['auc_rocs'].append(stat.auc_roc)

        f.write("\nPrompt Type Performance\n")
        f.write("-" * 20 + "\n")
        for prompt_type, stats in prompt_stats.items():
            f.write(f"\n{prompt_type}:\n")
            if stats['accuracies']:
                f.write(f"  Average Accuracy: {mean(stats['accuracies']):.2f}%\n")
            if stats['execution_times']:
                f.write(f"  Average Execution Time: {mean(stats['execution_times']):.4f}s\n")
            if stats['auc_rocs']:
                f.write(f"  Average AUC-ROC: {mean(stats['auc_rocs']):.4f}\n")
            f.write(f"  Total Timeouts: {sum(stats['timeout_counts'])}\n")