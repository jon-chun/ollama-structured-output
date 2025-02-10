# metrics.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TimeoutMetrics:
    """Records metrics related to timeout handling and retries"""
    occurred: bool = False
    retry_count: int = 0
    total_timeout_duration: float = 0.0
    final_timeout_duration: Optional[float] = None

@dataclass
class PromptMetrics:
    """Records metrics for a single prompt execution attempt"""
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