###CODE:
from ollama import chat
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import List, Dict, Union, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import os
import json
import random
import time
from datetime import datetime
from statistics import mean, median
import asyncio
import yaml
from pathlib import Path

# Enum for Prompt Types - Simplified from original code
class PromptType(str, Enum):
    SYSTEM1 = 'system1'
    COT = 'cot'

# Enum classes for predictions and risk weights
class Prediction(str, Enum):
    YES = 'YES'
    NO = 'NO'

    @classmethod
    def normalize_prediction(cls, value: str) -> 'Prediction':
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid prediction: {value}. Must be one of {list(cls.__members__.keys())}.")

class RiskWeight(str, Enum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

    @classmethod
    def normalize_weight(cls, value: str) -> 'RiskWeight':
        try:
            return cls[value.lower()]
        except KeyError:
            raise ValueError(f"Invalid risk weight: {value}. Must be one of {list(cls.__members__.keys())}.")

# Configuration management
class Config:
    """Global configuration singleton for managing application settings"""
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """Load configuration from config.yaml file"""
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Model-specific configuration parameters"""
        return self._config["model_config"]

    @property
    def execution(self) -> Dict[str, Any]:
        """Execution settings"""
        return self._config["execution"]

    @property
    def timeout(self) -> Dict[str, Any]:
        """Timeout configuration"""
        return self._config["timeout"]

    @property
    def logging(self) -> Dict[str, Any]:
        """Logging configuration"""
        return self._config["logging"]

    @property
    def output(self) -> Dict[str, Any]:
        """Output directory configuration"""
        return self._config["output"]

    @property
    def model_ensemble(self) -> Dict[str, Dict[str, Any]]:
        """Model ensemble configuration"""
        return self._config["model_ensemble"]

# Decision models using Pydantic
class RiskFactor(BaseModel):
    """Model for individual risk factors and their assessment"""
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=5)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionBase(BaseModel):
    """Base decision model with common fields"""
    prediction: Prediction
    confidence: int = Field(ge=0, le=100)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionCot(DecisionBase):
    """Extended decision model for chain-of-thought reasoning"""
    risk_factors: List[RiskFactor] = Field(..., min_items=1)

Decision = Union[DecisionBase, DecisionCot]

# Performance tracking classes
@dataclass
class TimeoutMetrics:
    """Metrics for tracking timeout behavior"""
    occurred: bool
    retry_count: int = 0
    final_timeout_duration: Optional[float] = None
    total_timeout_duration: float = 0

@dataclass
class PromptMetrics:
    """Detailed metrics for each prompt execution"""
    attempt_number: int
    execution_time_seconds: float
    successful: bool
    timeout_metrics: TimeoutMetrics
    error_message: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        """Ensure proper type conversion for metrics"""
        self.execution_time_seconds = float(self.execution_time_seconds)
        if self.prediction is not None:
            self.prediction = str(self.prediction)
        if self.confidence is not None:
            self.confidence = float(self.confidence)

class TimeoutStrategy:
    """
    Manages API timeout strategy with exponential backoff.
    This class provides sophisticated handling of API timeouts, including
    automatic retry logic and exponential backoff for better reliability.
    """
    def __init__(self, config: Config):
        # Initialize with configuration settings
        self.retry_count = 0
        self.base_timeout = config.timeout["max_api_wait_sec"]
        self.max_retries = config.timeout["max_api_timeout_retries"]
        self.step_increase = config.timeout["api_wait_step_increase_sec"]
        self.total_timeout_duration = 0
        self.attempt_durations = []

    def get_current_timeout(self) -> int:
        """
        Calculate timeout duration for current retry attempt using exponential backoff.
        Each retry increases the timeout duration to handle temporary network issues.
        """
        return self.base_timeout + (self.retry_count * self.step_increase)

    def should_retry(self) -> bool:
        """
        Determine if another retry attempt should be made based on configured limits.
        This prevents infinite retry loops while allowing for temporary issues to resolve.
        """
        return self.retry_count < self.max_retries

    def record_attempt(self, duration: float):
        """
        Record the duration of an attempt for metrics tracking.
        This helps in analyzing API performance patterns over time.
        """
        self.attempt_durations.append(duration)
        self.total_timeout_duration += duration

    def increment_retry(self):
        """
        Increment retry counter and update metrics.
        Includes logging for monitoring the retry process.
        """
        self.retry_count += 1
        current_timeout = self.get_current_timeout()
        logging.debug(
            f"Timeout retry #{self.retry_count} - "
            f"New timeout: {current_timeout}s"
        )

    def get_metrics(self) -> TimeoutMetrics:
        """
        Generate comprehensive timeout metrics for the current sequence.
        Provides detailed information about timeout behavior for analysis.
        """
        return TimeoutMetrics(
            occurred=self.retry_count > 0,
            retry_count=self.retry_count,
            final_timeout_duration=self.get_current_timeout(),
            total_timeout_duration=self.total_timeout_duration
        )

class ResponseProcessor:
    """Processes and validates model responses"""
    def __init__(self, prompt_type: PromptType):
        self.prompt_type = prompt_type
        self.decision_class = DecisionCot if prompt_type == PromptType.COT else DecisionBase

    def process_response(self, response_content: str) -> Optional[Decision]:
        """Process and validate the model's response"""
        try:
            decision = self.decision_class.model_validate_json(response_content)
            self._log_decision(decision)
            return decision
        except (ValidationError, Exception) as e:
            logging.error(f"Error processing response: {str(e)}")
            return None

    def _log_decision(self, decision: Decision):
        """Log the details of a decision"""
        logging.info(f"Prediction: {decision.prediction}")
        logging.info(f"Confidence: {decision.confidence}")
        
        if isinstance(decision, DecisionCot):
            for rf in decision.risk_factors:
                logging.info(f"Risk Factor: {rf.factor} ({rf.weight}): {rf.reasoning}")

@dataclass
class TimeoutStats:
    """Statistics related to timeout occurrences"""
    total_timeouts: int
    avg_timeout_duration: float
    max_timeout_duration: float
    total_timeout_duration: float

@dataclass
class PerformanceStats:
    """Complete statistics for a session"""
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
    timeout_stats: TimeoutStats

class PerformanceTracker:
    """Tracks and analyzes performance metrics"""
    def __init__(self, prompt_type: str, model_name: str):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.metrics: List[PromptMetrics] = []
        self.start_time = datetime.now()

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
        
        return TimeoutStats(
            total_timeouts=len(timeout_durations),
            avg_timeout_duration=mean(timeout_durations) if timeout_durations else 0.0,
            max_timeout_duration=max(timeout_durations) if timeout_durations else 0.0,
            total_timeout_duration=sum(timeout_durations)
        )

    def _generate_stats(self) -> PerformanceStats:
        """Generate comprehensive statistics from recorded metrics"""
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
            timeout_stats=self._calculate_timeout_stats()
        )

    def save_metrics(self, execution_time: float):
        """Save performance metrics to file"""
        stats = self._generate_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON format
        json_path = f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert the dataclass instance to a dictionary
            stats_dict = asdict(stats)
            json.dump(stats_dict, f, indent=2, default=str)
        
        # Save detailed text report
        self._save_text_report(stats, execution_time, timestamp)

    def _save_text_report(self, stats: PerformanceStats, execution_time: float, timestamp: str):
        """Save a human-readable performance report"""
        report_path = f"report_{self.model_name}_{self.prompt_type}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Performance Report - {self.model_name}\n")
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
            
            f.write("Timing Statistics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Execution: {stats.avg_execution_time:.2f}s\n")
            f.write(f"Median Execution: {stats.median_execution_time:.2f}s\n")
            
            if stats.timeout_stats.total_timeouts > 0:
                f.write("\nTimeout Statistics\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Timeouts: {stats.timeout_stats.total_timeouts}\n")
                f.write(f"Average Duration: {stats.timeout_stats.avg_timeout_duration:.2f}s\n")
                f.write(f"Maximum Duration: {stats.timeout_stats.max_timeout_duration:.2f}s\n")
                f.write(f"Total Duration: {stats.timeout_stats.total_timeout_duration:.2f}s\n")


@dataclass
class ApiCallMetadata:
    """Captures detailed information about each individual API call"""
    total_duration: float  # Total time for the API call
    prompt_tokens: int     # Number of tokens in the prompt
    completion_tokens: int # Number of tokens in the completion
    total_tokens: int     # Total tokens used
    first_token_time: float  # Time to first token
    tokens_per_second: float # Generation speed
    raw_response: Dict[str, Any]  # Raw response from the API for future analysis

@dataclass
class AggregateApiMetadata:
    """Aggregates API statistics across multiple calls"""
    total_calls: int
    total_duration: float
    avg_duration: float
    total_prompt_tokens: int
    avg_prompt_tokens: float
    total_completion_tokens: int
    avg_completion_tokens: float
    total_tokens: int
    avg_tokens: float
    avg_first_token_time: float
    avg_tokens_per_second: float
    token_distribution: Dict[str, Dict[str, float]]  # Statistics about token usage patterns

class ApiMetricsTracker:
    """Tracks and analyzes API usage metrics"""
    def __init__(self):
        self.calls: List[ApiCallMetadata] = []

    def record_call(self, metadata: ApiCallMetadata):
        """Record metrics for a single API call"""
        self.calls.append(metadata)

    def get_aggregate_metrics(self) -> AggregateApiMetadata:
        """Calculate aggregate statistics from all recorded calls"""
        if not self.calls:
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
                token_distribution={}
            )

        # Calculate basic aggregates
        total_calls = len(self.calls)
        total_duration = sum(call.total_duration for call in self.calls)
        total_prompt_tokens = sum(call.prompt_tokens for call in self.calls)
        total_completion_tokens = sum(call.completion_tokens for call in self.calls)
        total_tokens = total_prompt_tokens + total_completion_tokens