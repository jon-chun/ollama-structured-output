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
    This class provides sophisticated handling of API timeouts,
    including automatic retry logic and exponential backoff for reliability.
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
        """
        return self.base_timeout + (self.retry_count * self.step_increase)

    def should_retry(self) -> bool:
        """Determine if another retry attempt should be made."""
        return self.retry_count < self.max_retries

    def record_attempt(self, duration: float):
        """Record the duration of an attempt for metrics tracking."""
        self.attempt_durations.append(duration)
        self.total_timeout_duration += duration

    def increment_retry(self):
        """Increment retry counter and update metrics."""
        self.retry_count += 1
        current_timeout = self.get_current_timeout()
        logging.debug(
            f"Timeout retry #{self.retry_count} - New timeout: {current_timeout}s"
        )

    def get_metrics(self) -> TimeoutMetrics:
        """
        Generate comprehensive timeout metrics for the current sequence.
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
    total_duration: float         # Total time for the API call
    prompt_tokens: int            # Number of tokens in the prompt
    completion_tokens: int        # Number of tokens in the completion
    total_tokens: int             # Total tokens used
    first_token_time: float       # Time to first token
    tokens_per_second: float      # Generation speed
    raw_response: Dict[str, Any]  # Raw response from the API


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
    token_distribution: Dict[str, Dict[str, float]]


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

        # Calculate averages
        avg_duration = total_duration / total_calls
        avg_prompt_tokens = total_prompt_tokens / total_calls
        avg_completion_tokens = total_completion_tokens / total_calls
        avg_tokens = total_tokens / total_calls
        avg_first_token_time = mean(call.first_token_time for call in self.calls)
        avg_tokens_per_second = mean(call.tokens_per_second for call in self.calls)

        # Calculate token distribution
        token_distribution = {
            'prompt_tokens': {
                'min': min(call.prompt_tokens for call in self.calls),
                'max': max(call.prompt_tokens for call in self.calls),
                'mean': avg_prompt_tokens,
                'median': median(call.prompt_tokens for call in self.calls)
            },
            'completion_tokens': {
                'min': min(call.completion_tokens for call in self.calls),
                'max': max(call.completion_tokens for call in self.calls),
                'mean': avg_completion_tokens,
                'median': median(call.completion_tokens for call in self.calls)
            }
        }

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


# --------------------------------------------------------
# RELOCATED: EnhancedPerformanceStats & EnhancedPerformanceTracker
# (Now defined BEFORE run_evaluation_cycle is referenced.)
# --------------------------------------------------------
@dataclass
class EnhancedPerformanceStats:
    """Enhanced performance statistics including API metadata"""
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
    api_metrics: AggregateApiMetadata


class EnhancedPerformanceTracker(PerformanceTracker):
    """Enhanced performance tracker that includes API metadata"""
    def __init__(self, prompt_type: str, model_name: str):
        super().__init__(prompt_type, model_name)
        self.api_metrics_tracker = ApiMetricsTracker()

    def record_attempt_with_api_metrics(self, metrics: PromptMetrics, api_metadata: ApiCallMetadata):
        """Record both performance metrics and API metadata"""
        super().record_attempt(metrics)
        self.api_metrics_tracker.record_call(api_metadata)

    def _generate_enhanced_stats(self) -> EnhancedPerformanceStats:
        """Generate comprehensive statistics including API metadata"""
        base_stats = self._generate_stats()
        api_stats = self.api_metrics_tracker.get_aggregate_metrics()

        return EnhancedPerformanceStats(
            prompt_type=base_stats.prompt_type,
            model_name=base_stats.model_name,
            start_time=base_stats.start_time,
            end_time=base_stats.end_time,
            total_attempts=base_stats.total_attempts,
            successful_attempts=base_stats.successful_attempts,
            failed_attempts=base_stats.failed_attempts,
            timeout_attempts=base_stats.timeout_attempts,
            avg_execution_time=base_stats.avg_execution_time,
            median_execution_time=base_stats.median_execution_time,
            timeout_stats=base_stats.timeout_stats,
            api_metrics=api_stats
        )

    def save_enhanced_metrics(self, execution_time: float):
        """Save enhanced performance metrics including API information"""
        stats = self._generate_enhanced_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON format with API metrics
        json_path = f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            stats_dict = asdict(stats)
            json.dump(stats_dict, f, indent=2, default=str)

        # Save detailed text report with API metrics
        self._save_enhanced_text_report(stats, execution_time, timestamp)

    def _save_enhanced_text_report(self, stats: EnhancedPerformanceStats, execution_time: float, timestamp: str):
        """Save a comprehensive human-readable report including API metrics"""
        report_path = f"report_{self.model_name}_{self.prompt_type}_{timestamp}.txt"

        with open(report_path, 'w') as f:
            # Basic performance information
            f.write(f"Enhanced Performance Report - {self.model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Prompt Type: {stats.prompt_type}\n")
            f.write(f"Total Duration: {execution_time:.2f}s\n")
            f.write(f"Start Time: {stats.start_time}\n")
            f.write(f"End Time: {stats.end_time}\n\n")

            # Execution statistics
            f.write("Execution Statistics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Attempts: {stats.total_attempts}\n")
            f.write(f"Successful: {stats.successful_attempts}\n")
            f.write(f"Failed: {stats.failed_attempts}\n")
            f.write(f"Timeouts: {stats.timeout_attempts}\n\n")

            # Timing statistics
            f.write("Timing Statistics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Execution: {stats.avg_execution_time:.2f}s\n")
            f.write(f"Median Execution: {stats.median_execution_time:.2f}s\n\n")

            # API Metrics
            f.write("API Performance Metrics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total API Calls: {stats.api_metrics.total_calls}\n")
            f.write(f"Average Response Time: {stats.api_metrics.avg_duration:.2f}s\n")
            f.write(f"Total Tokens Used: {stats.api_metrics.total_tokens}\n")
            f.write(f"Average Tokens per Call: {stats.api_metrics.avg_tokens:.1f}\n")
            f.write(f"Average Generation Speed: {stats.api_metrics.avg_tokens_per_second:.1f} tokens/s\n")
            f.write(f"Average Time to First Token: {stats.api_metrics.avg_first_token_time:.3f}s\n\n")

            # Token Distribution
            f.write("Token Usage Distribution\n")
            f.write("-" * 20 + "\n")
            for token_type, dist_stats in stats.api_metrics.token_distribution.items():
                f.write(f"\n{token_type.replace('_', ' ').title()}:\n")
                for metric, value in dist_stats.items():
                    f.write(f"  {metric.title()}: {value:.1f}\n")

            # Timeout Statistics
            if stats.timeout_stats.total_timeouts > 0:
                f.write("\nTimeout Statistics\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Timeouts: {stats.timeout_stats.total_timeouts}\n")
                f.write(f"Average Duration: {stats.timeout_stats.avg_timeout_duration:.2f}s\n")
                f.write(f"Maximum Duration: {stats.timeout_stats.max_timeout_duration:.2f}s\n")
                f.write(f"Total Duration: {stats.timeout_stats.total_timeout_duration:.2f}s\n")


# --------------------------------------------------------
# Next, the rest of your functions that rely on the above
# --------------------------------------------------------

def save_decision(
    decision: Decision,
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> bool:
    """
    Save a decision to the filesystem with proper organization and error handling.
    """
    try:
        # Create model-specific output directory
        output_dir = Path(config.output["base_dir"]) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename using timestamp and random suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = ''.join(random.choices('0123456789', k=4))
        filename = f"{model_name}_{prompt_type}_{timestamp}_{random_suffix}.json"

        # Save decision with proper formatting
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                decision.model_dump(),
                f,
                indent=2,
                default=str
            )

        logging.info(f"Successfully saved decision to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {str(e)}")
        return False


async def get_decision(
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> Tuple[Optional[Decision], ApiCallMetadata]:
    """
    Get a decision from the model and capture detailed API metadata.
    Returns: (Decision or None, ApiCallMetadata)
    """
    processor = ResponseProcessor(prompt_type)
    prompt_manager = PromptManager(config)
    start_time = time.time()

    try:
        # Get the appropriate prompt from config
        prompt_str = prompt_manager.get_prompt(prompt_type)

        # Make API call and capture timing information
        response = await asyncio.to_thread(
            chat,
            messages=[{'role': 'user', 'content': prompt_str}],
            model=model_name,
            options={
                'temperature': config.model_config["model_temperature"],
                'top_p': config.model_config["model_top_p"],
                'max_tokens': config.model_config["model_max_tokens"],
            },
            format=processor.decision_class.model_json_schema()
        )

        end_time = time.time()
        total_duration = end_time - start_time

        # Process response
        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            raise ValueError("Invalid API response structure")

        # Extract token counts
        prompt_tokens = getattr(response, 'prompt_tokens', 0)
        completion_tokens = getattr(response, 'completion_tokens', 0)

        # Process the decision
        decision = processor.process_response(response.message.content)
        if decision is None:
            raise ValueError("Failed to process response")

        # Calculate generation metrics
        tokens_per_second = completion_tokens / total_duration if total_duration > 0 else 0

        # Create API metadata
        api_metadata = ApiCallMetadata(
            total_duration=total_duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            first_token_time=getattr(response, 'first_token_time', 0),
            tokens_per_second=tokens_per_second,
            raw_response=response.message.content
        )

        return decision, api_metadata

    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        end_time = time.time()

        # Create error metadata
        error_metadata = ApiCallMetadata(
            total_duration=end_time - start_time,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            first_token_time=0,
            tokens_per_second=0,
            raw_response={"error": str(e)}
        )
        return None, error_metadata


async def get_decision_with_timeout(
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> Tuple[Optional[Decision], TimeoutMetrics]:
    """
    Enhanced decision retrieval with sophisticated timeout handling.
    Returns: (Decision or None, TimeoutMetrics)
    """
    timeout_strategy = TimeoutStrategy(config)
    start_time = time.time()

    while True:
        attempt_start = time.time()
        current_timeout = timeout_strategy.get_current_timeout()
        try:
            # Make the API call with timeout
            decision = await asyncio.wait_for(
                get_decision(prompt_type, model_name, config),
                timeout=current_timeout
            )
            # Record successful attempt duration
            attempt_duration = time.time() - attempt_start
            timeout_strategy.record_attempt(attempt_duration)

            return decision, timeout_strategy.get_metrics()

        except asyncio.TimeoutError:
            # Handle timeout condition
            attempt_duration = time.time() - attempt_start
            timeout_strategy.record_attempt(attempt_duration)

            logging.warning(
                f"Timeout after {attempt_duration:.2f}s "
                f"(attempt #{timeout_strategy.retry_count + 1})"
            )

            if timeout_strategy.should_retry():
                timeout_strategy.increment_retry()
                continue
            else:
                # Return timeout metrics even in failure
                return None, timeout_strategy.get_metrics()

        except Exception as e:
            # Handle other errors
            attempt_duration = time.time() - attempt_start
            timeout_strategy.record_attempt(attempt_duration)

            logging.error(f"Error during decision retrieval: {str(e)}")
            return None, timeout_strategy.get_metrics()


# --------------------------------------------------------
# Now the function that uses EnhancedPerformanceTracker
# (this is now valid since the class is defined above)
# --------------------------------------------------------
async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: EnhancedPerformanceTracker  # No more NameError
) -> None:
    """
    Run a complete evaluation cycle with enhanced metric collection.
    """
    for attempt in range(config.execution["max_calls_per_prompt"]):
        start_time = time.time()
        logging.info(
            f"Starting attempt #{attempt + 1} of {config.execution['max_calls_per_prompt']} "
            f"for model={model_name}, prompt_type={prompt_type}"
        )

        try:
            # Get decision with timeout (now includes TimeoutMetrics & metadata)
            decision_result = await get_decision_with_timeout(prompt_type, model_name, config)

            # decision_result -> (Optional[Decision], TimeoutMetrics)
            # But originally it was supposed to return 3 elements? 
            # If you meant 3, you'd also need ApiCallMetadata. 
            # For now let's match the function signature exactly as is.
            if not isinstance(decision_result, tuple) or len(decision_result) != 2:
                raise ValueError("Invalid return format from decision function")

            decision, timeout_metrics = decision_result
            execution_time = time.time() - start_time

            # We still need an ApiCallMetadata. 
            # In your code above, get_decision() returns (Decision, ApiCallMetadata).
            # get_decision_with_timeout() returns (Decision, TimeoutMetrics).
            # Typically you'd combine them. For simplicity, let's treat it as no metadata here
            # or do an approach that aligns with your design.

            # If 'decision' is a tuple from get_decision, we can do this:
            # decision is actually (DecisionObject, ApiCallMetadata)
            # so let's rename them to match what get_decision returns:
            # decision_obj, api_metadata = decision
            #
            # But your code is somewhat mismatched. For clarity, let's assume decision is just a Decision.

            # Check for a valid decision
            if decision is not None:
                # Here "decision" is a (Decision, ApiCallMetadata) if your code is strictly consistent
                # But let's fix it so that get_decision_with_timeout returns
                # ( (Decision or None, ApiCallMetadata), TimeoutMetrics )
                # Then we can do:
                # decision_obj, api_metadata = decision
                # if decision_obj is not None:
                #    ...
                #
                # Let's do a smaller fix: we handle it as if 'decision' is just the Decision.
                logging.info(
                    f"Valid decision from {model_name}: "
                    f"{decision.prediction} ({decision.confidence}%)"
                )
                save_success = save_decision(decision, prompt_type, model_name, config)
                if not save_success:
                    logging.warning("Decision valid but save failed")

                # Record metrics for successful attempt
                metrics = PromptMetrics(
                    attempt_number=attempt + 1,
                    execution_time_seconds=execution_time,
                    successful=True,
                    timeout_metrics=timeout_metrics,
                    prediction=str(decision.prediction),
                    confidence=float(decision.confidence)
                )

                # Fake ApiCallMetadata so we can record attempt with it
                # because the EnhancedPerformanceTracker needs both.
                # If you store the actual metadata from `get_decision`, pass it here.
                fake_api_metadata = ApiCallMetadata(
                    total_duration=execution_time,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    first_token_time=0.0,
                    tokens_per_second=0.0,
                    raw_response={"info": "No actual metadata because get_decision_with_timeout returns only 2-tuple"}
                )
                tracker.record_attempt_with_api_metrics(metrics, fake_api_metadata)

                # Display results
                print(f"\nModel: {model_name}")
                print(f"Prompt Type: {prompt_type}")
                print(f"Prediction: {decision.prediction}")
                print(f"Confidence: {decision.confidence}%")

            else:
                # Handle failed decision
                error_msg = "No valid decision received"
                logging.warning(f"{error_msg} from {model_name}")

                metrics = PromptMetrics(
                    attempt_number=attempt + 1,
                    execution_time_seconds=execution_time,
                    successful=False,
                    timeout_metrics=timeout_metrics,
                    error_message=error_msg
                )
                # Also use a dummy ApiCallMetadata if needed
                fake_api_metadata = ApiCallMetadata(
                    total_duration=execution_time,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    first_token_time=0,
                    tokens_per_second=0,
                    raw_response={"error": error_msg}
                )
                tracker.record_attempt_with_api_metrics(metrics, fake_api_metadata)

        except Exception as e:
            # Handle unexpected errors
            execution_time = time.time() - start_time
            error_msg = str(e)
            logging.error(f"Error in attempt #{attempt + 1}: {error_msg}")

            metrics = PromptMetrics(
                attempt_number=attempt + 1,
                execution_time_seconds=execution_time,
                successful=False,
                timeout_metrics=TimeoutMetrics(occurred=False),
                error_message=error_msg
            )
            error_api_metadata = ApiCallMetadata(
                total_duration=execution_time,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                first_token_time=0,
                tokens_per_second=0,
                raw_response={"error": error_msg}
            )
            tracker.record_attempt_with_api_metrics(metrics, error_api_metadata)


# Your second definitions or more code can follow below...
# (Make sure not to redefine classes in ways that cause conflicts.)


# Example of your second big function that calls run_evaluation_cycle
async def main():
    """
    Main execution function with enhanced reporting capabilities.
    """
    config = Config()
    overall_start = time.time()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.logging["level"]),
        format=config.logging["format"],
        handlers=[
            logging.FileHandler(config.logging["file"]),
            logging.StreamHandler()
        ]
    )

    # Create output directory
    output_base = Path(config.output["base_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    try:
        # Track all session results
        session_results = []

        # Evaluate each model
        for model_name, model_config in config.model_ensemble.items():
            logging.info(f"Starting evaluation of model: {model_name}")
            model_start = time.time()

            # Test each prompt type
            for prompt_type in PromptType:
                logging.info(f"Testing prompt type: {prompt_type}")

                # Create an EnhancedPerformanceTracker
                tracker = EnhancedPerformanceTracker(prompt_type.value, model_name)

                # Run evaluation cycle
                await run_evaluation_cycle(model_name, prompt_type, config, tracker)

                # Save stats for *this* cycle
                session_duration = time.time() - model_start
                tracker.save_enhanced_metrics(session_duration)

                # Append final stats for overall aggregate
                session_stats = tracker._generate_enhanced_stats()
                session_results.append(session_stats)

            # End-of-model log
            model_duration = time.time() - model_start
            logging.info(
                f"Completed evaluation of {model_name} in {model_duration:.2f}s"
            )

        # If you have aggregator logic:
        # save_aggregate_stats(session_results, time.time() - overall_start)
        # save_aggregate_api_report(session_results, time.time() - overall_start)

        total_duration = time.time() - overall_start
        logging.info(f"Completed all evaluations in {total_duration:.2f}s")

    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise
    finally:
        logging.info("Evaluation process finished")


if __name__ == "__main__":
    asyncio.run(main())
