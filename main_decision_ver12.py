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

async def get_decision(
    prompt_type: PromptType, 
    model_name: str, 
    config: Config
) -> Optional[Decision]:
    """
    Get a decision from the model without timeout handling.
    This function focuses solely on making the API call and processing the response.
    
    Args:
        prompt_type: Type of prompt being used
        model_name: Name of the model to query
        config: Configuration settings
        
    Returns:
        Decision object if successful, None otherwise
    """
    processor = ResponseProcessor(prompt_type)
    
    try:
        # Get the appropriate prompt based on type
        prompt_str = PROMPT_SYSTEM1 if prompt_type == PromptType.SYSTEM1 else PROMPT_COT
        
        # Make API call
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
        
        # Process response
        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            raise ValueError("Invalid API response structure")
        
        decision = processor.process_response(response.message.content)
        if decision is None:
            raise ValueError("Failed to process response")
        
        return decision
            
    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None

async def get_decision_with_timeout(
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> Tuple[Optional[Decision], TimeoutMetrics]:
    """
    Enhanced decision retrieval with sophisticated timeout handling.
    This function wraps get_decision() with timeout functionality.
    
    Args:
        prompt_type: Type of prompt being used
        model_name: Name of the model to query
        config: Configuration settings
        
    Returns:
        A tuple containing:
        - The Decision object if successful, None otherwise
        - TimeoutMetrics tracking timing and retry information
    """
    timeout_strategy = TimeoutStrategy(config)
    start_time = time.time()
    
    while True:
        attempt_start = time.time()
        current_timeout = timeout_strategy.get_current_timeout()
        
        try:
            # Make the API call with timeout protection
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
                # Return timeout metrics even in failure case
                return None, timeout_strategy.get_metrics()
        
        except Exception as e:
            # Handle other types of errors
            attempt_duration = time.time() - attempt_start
            timeout_strategy.record_attempt(attempt_duration)
            
            logging.error(f"Error during decision retrieval: {str(e)}")
            return None, timeout_strategy.get_metrics()
        
def save_decision(
    decision: Decision,
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> bool:
    """
    Save a decision to the filesystem with proper organization and error handling.
    
    Arguments:
        decision: The validated decision object
        prompt_type: Type of prompt used to generate the decision
        model_name: Name of the model that generated the decision
        config: Configuration object containing output settings
    
    Returns:
        bool: True if save was successful, False otherwise
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

async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker
) -> None:
    """
    Run a complete evaluation cycle for a specific model and prompt type combination.
    This function manages the evaluation process, including metrics collection and error handling.
    
    Args:
        model_name: Name of the model to evaluate
        prompt_type: Type of prompt to use
        config: Configuration object
        tracker: Performance tracker instance
    """
    for attempt in range(config.execution["max_calls_per_prompt"]):
        start_time = time.time()
        logging.info(
            f"Starting attempt #{attempt + 1} of {config.execution['max_calls_per_prompt']} "
            f"for model={model_name}, prompt_type={prompt_type}"
        )
        
        try:
            # Get decision with timeout handling - returns (Decision, TimeoutMetrics)
            decision_result = await get_decision_with_timeout(
                prompt_type=prompt_type,
                model_name=model_name,
                config=config
            )
            
            # Unpack the result tuple
            if not isinstance(decision_result, tuple) or len(decision_result) != 2:
                raise ValueError("Invalid return format from decision function")
            
            decision, timeout_metrics = decision_result
            execution_time = time.time() - start_time
            
            if decision is not None:
                # Process successful decision
                logging.info(
                    f"Valid decision from {model_name}: "
                    f"{decision.prediction} ({decision.confidence}%)"
                )
                
                # Save the decision
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
                
                # Display results
                print(f"\nModel: {model_name}")
                print(f"Prompt Type: {prompt_type}")
                print(f"Prediction: {decision.prediction}")
                print(f"Confidence: {decision.confidence}%")
                
                # For chain-of-thought decisions, show additional analysis
                if isinstance(decision, DecisionCot):
                    print("\nRisk Factor Analysis:")
                    for rf in decision.risk_factors:
                        print(f"- {rf.factor} ({rf.weight}): {rf.reasoning}")
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
        
        # Record metrics for this attempt
        tracker.record_attempt(metrics)
        logging.info(
            f"Attempt #{attempt + 1} completed in {execution_time:.2f}s "
            f"(Status: {'Success' if metrics.successful else 'Failed'})"
        )

async def run_evaluation_session(
    model_name: str,
    prompt_type: PromptType,
    config: Config
) -> Optional[PerformanceStats]:
    """
    Runs a complete evaluation session for a specific model and prompt type.
    
    This function manages the entire process of evaluating a model with a specific
    prompt type, including multiple attempts, performance tracking, and proper
    error handling. It provides comprehensive statistics about the session.
    
    Args:
        model_name: Name of the model to evaluate
        prompt_type: Type of prompt to use
        config: Configuration settings
        
    Returns:
        PerformanceStats if successful, None if critical failure
    """
    tracker = PerformanceTracker(prompt_type, model_name)
    session_start = time.time()
    
    try:
        for attempt in range(config.execution["max_calls_per_prompt"]):
            attempt_start = time.time()
            logging.info(
                f"Starting attempt #{attempt + 1} of "
                f"{config.execution['max_calls_per_prompt']} for "
                f"model={model_name}, prompt_type={prompt_type}"
            )
            
            try:
                # Get decision with timeout handling
                decision, timeout_metrics = await get_decision_with_timeout(
                    prompt_type=prompt_type,
                    model_name=model_name,
                    config=config
                )
                
                execution_time = time.time() - attempt_start
                
                if decision is not None:
                    # Handle successful decision
                    logging.info(
                        f"Valid decision from {model_name}: "
                        f"{decision.prediction} ({decision.confidence}%)"
                    )
                    
                    # Save decision to filesystem
                    save_success = save_decision(
                        decision, prompt_type, model_name, config
                    )
                    if not save_success:
                        logging.warning("Decision valid but save failed")
                    
                    # Record successful metrics
                    metrics = PromptMetrics(
                        attempt_number=attempt + 1,
                        execution_time_seconds=execution_time,
                        successful=True,
                        timeout_metrics=timeout_metrics,
                        prediction=str(decision.prediction),
                        confidence=float(decision.confidence)
                    )
                    
                    # Display results
                    print(f"\nModel: {model_name}")
                    print(f"Prompt Type: {prompt_type}")
                    print(f"Prediction: {decision.prediction}")
                    print(f"Confidence: {decision.confidence}%")
                    
                else:
                    # Handle failed attempt
                    metrics = PromptMetrics(
                        attempt_number=attempt + 1,
                        execution_time_seconds=execution_time,
                        successful=False,
                        timeout_metrics=timeout_metrics,
                        error_message="No valid decision received"
                    )
            
            except Exception as e:
                # Handle unexpected errors
                execution_time = time.time() - attempt_start
                logging.error(f"Error in attempt #{attempt + 1}: {str(e)}")
                
                metrics = PromptMetrics(
                    attempt_number=attempt + 1,
                    execution_time_seconds=execution_time,
                    successful=False,
                    timeout_metrics=TimeoutMetrics(occurred=False),
                    error_message=str(e)
                )
            
            # Record metrics for this attempt
            tracker.record_attempt(metrics)
            logging.info(
                f"Attempt #{attempt + 1} completed in {execution_time:.2f}s "
                f"(Status: {'Success' if metrics.successful else 'Failed'})"
            )
        
        # Generate and save session statistics
        session_duration = time.time() - session_start
        tracker.save_metrics(session_duration)
        
        return tracker._generate_stats()
        
    except Exception as e:
        logging.error(f"Critical error in evaluation session: {str(e)}")
        return None


@dataclass
class AggregateStats:
    """Aggregate statistics across all evaluation sessions"""
    total_duration: float
    total_sessions: int
    sessions_by_model: Dict[str, int]
    sessions_by_prompt: Dict[str, int]
    overall_success_rate: float
    avg_execution_time: float
    total_timeouts: int
    avg_timeout_duration: float
    model_performance: Dict[str, Dict[str, float]]
    prompt_performance: Dict[str, Dict[str, float]]

def calculate_aggregate_stats(
    session_results: List[PerformanceStats],
    total_duration: float
) -> AggregateStats:
    """
    Calculate aggregate statistics across all evaluation sessions.
    This provides a high-level view of system performance and helps
    identify patterns across different models and prompt types.
    """
    # Initialize counters
    total_successes = 0
    total_attempts = 0
    total_timeouts = 0
    total_timeout_duration = 0
    execution_times = []
    sessions_by_model = {}
    sessions_by_prompt = {}
    model_performance = {}
    prompt_performance = {}
    
    # Process each session
    for session in session_results:
        # Update session counts
        sessions_by_model[session.model_name] = \
            sessions_by_model.get(session.model_name, 0) + 1
        sessions_by_prompt[session.prompt_type] = \
            sessions_by_prompt.get(session.prompt_type, 0) + 1
        
        # Update success metrics
        total_successes += session.successful_attempts
        total_attempts += session.total_attempts
        total_timeouts += session.timeout_attempts
        
        # Update timing metrics
        execution_times.append(session.avg_execution_time)
        total_timeout_duration += session.timeout_stats.total_timeout_duration
        
        # Update model performance metrics
        if session.model_name not in model_performance:
            model_performance[session.model_name] = {
                'success_rate': 0,
                'avg_execution_time': 0,
                'timeout_rate': 0
            }
        
        model_stats = model_performance[session.model_name]
        model_stats['success_rate'] = (session.successful_attempts / 
                                     session.total_attempts * 100)
        model_stats['avg_execution_time'] = session.avg_execution_time
        model_stats['timeout_rate'] = (session.timeout_attempts / 
                                     session.total_attempts * 100)
        
        # Update prompt performance metrics
        if session.prompt_type not in prompt_performance:
            prompt_performance[session.prompt_type] = {
                'success_rate': 0,
                'avg_execution_time': 0,
                'timeout_rate': 0
            }
        
        prompt_stats = prompt_performance[session.prompt_type]
        prompt_stats['success_rate'] = (session.successful_attempts / 
                                      session.total_attempts * 100)
        prompt_stats['avg_execution_time'] = session.avg_execution_time
        prompt_stats['timeout_rate'] = (session.timeout_attempts / 
                                      session.total_attempts * 100)
    
    return AggregateStats(
        total_duration=total_duration,
        total_sessions=len(session_results),
        sessions_by_model=sessions_by_model,
        sessions_by_prompt=sessions_by_prompt,
        overall_success_rate=(total_successes / total_attempts * 100 
                            if total_attempts > 0 else 0),
        avg_execution_time=mean(execution_times) if execution_times else 0,
        total_timeouts=total_timeouts,
        avg_timeout_duration=(total_timeout_duration / total_timeouts 
                            if total_timeouts > 0 else 0),
        model_performance=model_performance,
        prompt_performance=prompt_performance
    )

def save_aggregate_stats(session_results: List[PerformanceStats], 
                        total_duration: float):
    """
    Save aggregate statistics to both JSON and human-readable formats.
    This creates comprehensive reports that can be used for both
    automated analysis and human review.
    """
    # Calculate aggregate statistics
    stats = calculate_aggregate_stats(session_results, total_duration)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON format
    json_path = f"aggregate_stats_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(asdict(stats), f, indent=2, default=str)
    
    # Save human-readable report
    report_path = f"aggregate_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("Aggregate Performance Report\n")
        f.write("==========================\n\n")
        
        f.write("Overall Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total Duration: {stats.total_duration:.2f}s\n")
        f.write(f"Total Sessions: {stats.total_sessions}\n")
        f.write(f"Overall Success Rate: {stats.overall_success_rate:.2f}%\n")
        f.write(f"Average Execution Time: {stats.avg_execution_time:.2f}s\n")
        f.write(f"Total Timeouts: {stats.total_timeouts}\n")
        f.write(f"Average Timeout Duration: {stats.avg_timeout_duration:.2f}s\n\n")
        
        f.write("Model Performance\n")
        f.write("----------------\n")
        for model, stats in stats.model_performance.items():
            f.write(f"\n{model}:\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2f}%\n")
            f.write(f"  Avg Execution Time: {stats['avg_execution_time']:.2f}s\n")
            f.write(f"  Timeout Rate: {stats['timeout_rate']:.2f}%\n")
        
        f.write("\nPrompt Performance\n")
        f.write("-----------------\n")
        for prompt, stats in stats.prompt_performance.items():
            f.write(f"\n{prompt}:\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2f}%\n")
            f.write(f"  Avg Execution Time: {stats['avg_execution_time']:.2f}s\n")
            f.write(f"  Timeout Rate: {stats['timeout_rate']:.2f}%\n")

class PromptManager:
    """Manages prompt texts and their configuration"""
    def __init__(self, config: Config):
        self.config = config
        self._prompts = {
            PromptType.SYSTEM1: PROMPT_SYSTEM1,
            PromptType.COT: PROMPT_COT
        }
    
    def get_prompt(self, prompt_type: PromptType) -> str:
        """Get the appropriate prompt text for the given type"""
        return self._prompts.get(prompt_type)
    
    def update_prompt(self, prompt_type: PromptType, new_prompt: str):
        """Update the prompt text for a given type"""
        self._prompts[prompt_type] = new_prompt
        logging.info(f"Updated prompt text for {prompt_type}")

async def main():
    """
    Main execution function coordinating the entire evaluation process.
    
    This function manages the complete workflow of evaluating multiple models
    with different prompt types. It includes proper error handling, logging,
    and performance tracking throughout the process.
    """
    # Initialize configuration and timing
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
                
                # Run evaluation session
                session_stats = await run_evaluation_session(
                    model_name, prompt_type, config
                )
                
                if session_stats:
                    session_results.append(session_stats)
            
            # Log model completion
            model_duration = time.time() - model_start
            logging.info(
                f"Completed evaluation of {model_name} "
                f"in {model_duration:.2f}s"
            )
        
        # Generate and save aggregate statistics
        if session_results:
            save_aggregate_stats(session_results, time.time() - overall_start)
        
        # Log overall completion
        total_duration = time.time() - overall_start
        logging.info(f"Completed all evaluations in {total_duration:.2f}s")
        
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise
    
    finally:
        logging.info("Evaluation process finished")

# Define prompt texts
PROMPT_SYSTEM1 = """
###TASK: 
Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

###RISK_FACTORS:
The person is a 22-year-old male who identifies himself as White. He has been arrested 2 times and has been convicted. He is 5 feet 6 inches and weighs 120 lbs. He has never been married and is currently living alone. He does not have a high school diploma or GED certificate and is not enrolled in college. He has held 3 jobs during the year. He has used drugs in the past four years (marijuana and cocaine) and some of the time during the past month he has felt so down in the dumps that nothing could cheer him up. At a young age, he lived with his biological dad in a household of size 4, in a rural area. He considers that God has nothing to do with what happens to him personally.

###RESPONSE_FORMAT:
1. 'YES' or 'NO' (prediction)
2. integer 0-100 (confidence in percent)
"""

PROMPT_COT = """
###TASK:
Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

###RISK_FACTORS:
The person is an 18-year-old male who identifies himself as White. He has been arrested 10 times and has been convicted. He is 6 feet 1 inch and weighs 195 lbs. He has never been married and is currently living alone. He has a GED certificate and is not enrolled in college. He has held 1 job during the year. He has used drugs in the past four years (marijuana and cocaine) and has not felt depressed over the past month. At a young age, he lived with both of his biological parents in a household of size 4, in an urban area. He considers that God has something to do with what happens to him personally.

###RESPONSE_FORMAT:
1. 'YES' or 'NO' (prediction)
2. integer 0-100 (confidence in percent)
3. A list of text strings identifying each risk factor with reasoning to weight each risk factor as 'high', 'medium' or 'low'
"""

if __name__ == "__main__":
    asyncio.run(main())