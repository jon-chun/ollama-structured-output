from ollama import chat
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, asdict
import logging
import os
import json
import time
from datetime import datetime
from statistics import mean, median
import asyncio

# Model hyperparameters
MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.9
MODEL_MAX_TOKENS = 1024

# Prompt calls and retries
MAX_CALLS_PER_PROMPT = 10
MAX_RETRIES_MALFORMED = 3

# API timeout configuration
MAX_API_WAIT_SEC = 240  # Base maximum timeout for API calls
MAX_API_TIMEOUT_RETRIES = 3  # Maximum number of retries for timeout failures
API_WAIT_STEP_INCREASE_SEC = 30  # Increment for timeout duration per retry

PROMPT_TYPE = ''  # Defined here, but set in main

# Set up logging for better error tracking
logging.basicConfig(
    level=logging.DEBUG, # or .INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('decision_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


OLLAMA_MODEL = 'llama3.2:1b-instruct-q4_K_M' 
# OLLAMA_MODEL = 'llama3.1:8b-instruct-q4_K_M'  # Replace with actual model if necessary
# OLLAMA_MODEL = 'llama3.3:70b-instruct-q4_K_M'

# Prompt texts
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

# Enum for Prompt Types
class PromptType(str, Enum):
    SYSTEM1 = 'system1'
    COT = 'cot'
    COT_NSHOT = 'cot_nshot'

# Enum for Predictions
class Prediction(str, Enum):
    YES = 'YES'
    NO = 'NO'

    @classmethod
    def normalize_prediction(cls, value: str) -> 'Prediction':
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid prediction value: {value}. Must be one of {list(cls.__members__.keys())}."
            )

# Enum for Risk Weights
class RiskWeight(str, Enum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

    @classmethod
    def normalize_weight(cls, value: str) -> 'RiskWeight':
        try:
            return cls[value.lower()]
        except KeyError:
            raise ValueError(
                f"Invalid risk weight: {value}. Must be one of {list(cls.__members__.keys())}."
            )

# Pydantic Model for Risk Factors
class RiskFactor(BaseModel):
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=5)

    class Config:
        frozen = True
        extra = 'forbid'

# Decision Models
class DecisionSystem1(BaseModel):
    prediction: Prediction
    confidence: int = Field(ge=0, le=100)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionCot(DecisionSystem1):
    risk_factors: List[RiskFactor] = Field(..., min_items=1)

# Add type alias for Decision
Decision = Union[DecisionSystem1, DecisionCot]

# Response Processor
class ResponseProcessor:
    def __init__(self, prompt_type: PromptType):
        self.prompt_type = prompt_type
        # Use DecisionSystem1 if system1, otherwise DecisionCot for cot/cot_nshot
        if self.prompt_type == PromptType.SYSTEM1:
            self.decision_class = DecisionSystem1
        else:
            self.decision_class = DecisionCot

    def process_response(self, response_content: str) -> Optional[BaseModel]:
        try:
            decision = self.decision_class.model_validate_json(response_content)
            self._log_decision(decision)
            return decision
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing response: {e}")
            return None

    def _log_decision(self, decision: BaseModel) -> None:
        logger.info(f"Prediction: {decision.prediction}")
        logger.info(f"Confidence: {decision.confidence}")

        if isinstance(decision, DecisionCot):
            for rf in decision.risk_factors:
                logger.info(f"Risk Factor: {rf.factor} ({rf.weight}): {rf.reasoning}")

# 1) Function to check if the response is complete
def check_response_complete(prompt_type: PromptType, response: BaseModel, model: str) -> bool:
    """
    Checks if the response object is complete based on the prompt_type:
      - system1 must have 'prediction' (YES/NO) and 'confidence' (0-100)
      - cot or cot_nshot must have the above plus 'risk_factors' (list)
    Returns True if complete, otherwise False.
    """
    # Must have valid prediction and confidence in all cases
    if not hasattr(response, 'prediction') or not hasattr(response, 'confidence'):
        return False

    # Check prediction
    if response.prediction not in [Prediction.YES, Prediction.NO]:
        return False

    # Check confidence
    if not (0 <= response.confidence <= 100):
        return False

    # For 'cot' or 'cot_nshot', must also have risk_factors
    if prompt_type in [PromptType.COT, PromptType.COT_NSHOT]:
        if not hasattr(response, 'risk_factors'):
            return False
        if not response.risk_factors or len(response.risk_factors) == 0:
            return False

    return True

# Main Function
def get_decision(prompt_type: PromptType, model: str = OLLAMA_MODEL) -> Optional[BaseModel]:
    """
    Retrieves a decision from the chat endpoint, ensuring completeness up to MAX_RETRIES_MALFORMED times.
    If the response is complete, returns the validated model object; otherwise returns None.
    """
    processor = ResponseProcessor(prompt_type)
    prompt_str = PROMPT_SYSTEM1 if prompt_type == PromptType.SYSTEM1 else PROMPT_COT

    # 2) Loop until the response is complete or we reach MAX_RETRIES_MALFORMED
    for attempt in range(MAX_RETRIES_MALFORMED):
        logger.info(f"Retry #{attempt + 1} of {MAX_RETRIES_MALFORMED} for prompt_type={prompt_type}")

        try:
            response = chat(
                messages=[{'role': 'user', 'content': prompt_str}],
                model=model,
                options={
                    'temperature': MODEL_TEMPERATURE,
                    'top_p': MODEL_TOP_P,
                    'max_tokens': MODEL_MAX_TOKENS,
                },
                format=processor.decision_class.model_json_schema(),
            )

            # Validate structure of the API response
            if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
                logger.error("Invalid API response structure.")
                continue

            # Convert to domain model
            decision = processor.process_response(response.message.content)
            if decision is None:
                # Malformed response, try again
                continue

            # 2) Check completeness of the response
            if check_response_complete(prompt_type, decision, model):
                # 3) If complete, save JSON and return
                save_decision_json(decision, prompt_type, model)
                return decision
            else:
                logger.error("Response missing required fields.")
        except Exception as e:
            logger.error(f"Error during API call or processing: {e}")

    # If we exhaust all retries, return None
    return None

# 3) Export response object to JSON
def save_decision_json(decision: BaseModel, prompt_type: PromptType, model_name: str) -> None:
    """
    Saves the decision object as JSON in ./data/judgements/<model_name>/
    """
    # Create directory if not exists
    output_dir = os.path.join('.', 'data', 'judgements', model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Construct output filename
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_name}_{prompt_type}_{datetime_str}.json"
    output_path = os.path.join(output_dir, output_filename)

    # Convert decision Pydantic model to dict, then save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(decision.model_dump(), f, indent=2)
    logger.info(f"Saved decision to {output_path}")

@dataclass
class PromptPerformanceMetrics:
    """Tracks performance metrics for a single prompt execution"""
    attempt_number: int
    execution_time_seconds: float
    successful: bool
    error_message: str = None
    prediction: str = None
    confidence: float = None
    
@dataclass
class SessionPerformanceStats:
    """Aggregates performance metrics for an entire session"""
    prompt_type: str
    start_time: datetime
    end_time: datetime
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    avg_execution_time: float
    median_execution_time: float
    retry_counts: Dict[int, int]  # Maps attempt number to count of occurrences
    error_types: Dict[str, int]   # Maps error types to frequency








class TimeoutStrategy:
    """Manages API timeout strategy with exponential backoff"""
    def __init__(self):
        self.retry_count = 0
        self.base_timeout = MAX_API_WAIT_SEC
        
    def get_current_timeout(self) -> int:
        """Calculate timeout duration for current retry attempt"""
        return self.base_timeout + (self.retry_count * API_WAIT_STEP_INCREASE_SEC)
    
    def should_retry(self) -> bool:
        """Determine if another retry attempt should be made"""
        return self.retry_count < MAX_API_TIMEOUT_RETRIES
    
    def increment_retry(self):
        """Increment retry counter and log the action"""
        self.retry_count += 1
        logger.debug(
            f"Timeout retry #{self.retry_count} - "
            f"New timeout: {self.get_current_timeout()}s"
        )

@dataclass
class TimeoutMetrics:
    """Tracks timeout-related metrics for a single execution"""
    occurred: bool
    retry_count: int = 0
    final_timeout_duration: Optional[float] = None
    total_timeout_duration: Optional[float] = 0  # Sum of all timeout durations in retry sequence

@dataclass
class PromptPerformanceMetrics:
    """Tracks performance metrics for a single prompt execution"""
    attempt_number: int
    execution_time_seconds: float
    successful: bool
    timeout_metrics: TimeoutMetrics
    error_message: str = None
    prediction: str = None
    confidence: float = None

@dataclass
class SessionPerformanceStats:
    """Aggregates performance metrics for an entire session"""
    prompt_type: str
    start_time: datetime
    end_time: datetime
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    timeout_attempts: int
    avg_execution_time: float
    median_execution_time: float
    retry_counts: Dict[int, int]
    error_types: Dict[str, int]
    timeout_stats: Dict[str, float]  # Additional timeout statistics

class PerformanceTracker:
    def __init__(self, prompt_type: str):
        self.prompt_type = prompt_type
        self.metrics: List[PromptPerformanceMetrics] = []
        self.start_time = datetime.now()
        
    def record_attempt(self, metrics: PromptPerformanceMetrics):
        """Record metrics for a single attempt"""
        self.metrics.append(metrics)
        
        # Log performance information
        status = "successful" if metrics.successful else "failed"
        timeout_info = ""
        if metrics.timeout_metrics.occurred:
            timeout_info = (
                f" (timeout occurred, {metrics.timeout_metrics.retry_count} retries, "
                f"total timeout duration: {metrics.timeout_metrics.total_timeout_duration:.2f}s)"
            )
            
        logger.debug(
            f"Attempt #{metrics.attempt_number} {status} - "
            f"Execution time: {metrics.execution_time_seconds:.2f}s{timeout_info}"
        )
        
        if not metrics.successful:
            logger.error(f"Error in attempt #{metrics.attempt_number}: {metrics.error_message}")
            
    def generate_session_stats(self) -> SessionPerformanceStats:
        """Generate aggregate statistics for the session"""
        execution_times = [m.execution_time_seconds for m in self.metrics]
        retry_counts = {}
        error_types = {}
        timeout_attempts = sum(1 for m in self.metrics if m.timeout_metrics.occurred)
        
        # Calculate timeout-specific statistics
        timeout_stats = {
            "total_timeout_attempts": timeout_attempts,
            "avg_timeout_duration": mean([m.timeout_metrics.total_timeout_duration 
                                       for m in self.metrics if m.timeout_metrics.occurred] or [0]),
            "max_timeout_duration": max([m.timeout_metrics.total_timeout_duration 
                                      for m in self.metrics if m.timeout_metrics.occurred] or [0]),
            "total_timeout_duration": sum(m.timeout_metrics.total_timeout_duration 
                                        for m in self.metrics if m.timeout_metrics.occurred)
        }
        
        for metric in self.metrics:
            retry_counts[metric.attempt_number] = retry_counts.get(metric.attempt_number, 0) + 1
            if not metric.successful:
                error_type = type(metric.error_message).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return SessionPerformanceStats(
            prompt_type=self.prompt_type,
            start_time=self.start_time,
            end_time=datetime.now(),
            total_attempts=len(self.metrics),
            successful_attempts=sum(1 for m in self.metrics if m.successful),
            failed_attempts=sum(1 for m in self.metrics if not m.successful),
            timeout_attempts=timeout_attempts,
            avg_execution_time=mean(execution_times),
            median_execution_time=median(execution_times),
            retry_counts=retry_counts,
            error_types=error_types,
            timeout_stats=timeout_stats
        )
    
    def save_reports(self):
        """Generate and save both detailed text report and JSON report"""
        stats = self.generate_session_stats()
        
        # Save JSON report
        json_report = asdict(stats)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename_root = f"report_performance_stats_{OLLAMA_MODEL}_{self.prompt_type}_{datetime_str}"
        report_filename_json = f"{report_filename_root}.json"
        with open(report_filename_json, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Save detailed text report
        report_filename_txt = f"{report_filename_root}.txt"
        with open(report_filename_txt, 'w') as f:
            f.write("Performance Analysis Report\n")
            f.write("=========================\n\n")
            
            f.write(f"Prompt Type: {stats.prompt_type}\n")
            f.write(f"Session Duration: {stats.end_time - stats.start_time}\n")
            f.write(f"Total Attempts: {stats.total_attempts}\n")
            f.write(f"Successful Attempts: {stats.successful_attempts}\n")
            f.write(f"Failed Attempts: {stats.failed_attempts}\n")
            f.write(f"Timeout Attempts: {stats.timeout_attempts}\n\n")
            
            f.write("Timing Statistics\n")
            f.write("-----------------\n")
            f.write(f"Average Execution Time: {stats.avg_execution_time:.2f}s\n")
            f.write(f"Median Execution Time: {stats.median_execution_time:.2f}s\n\n")
            
            f.write("Timeout Statistics\n")
            f.write("-----------------\n")
            f.write(f"Total Timeout Attempts: {stats.timeout_stats['total_timeout_attempts']}\n")
            f.write(f"Average Timeout Duration: {stats.timeout_stats['avg_timeout_duration']:.2f}s\n")
            f.write(f"Maximum Timeout Duration: {stats.timeout_stats['max_timeout_duration']:.2f}s\n")
            f.write(f"Total Timeout Duration: {stats.timeout_stats['total_timeout_duration']:.2f}s\n\n")
            
            f.write("Retry Distribution\n")
            f.write("-----------------\n")
            for attempt_num, count in sorted(stats.retry_counts.items()):
                f.write(f"Attempt #{attempt_num}: {count} occurrences\n")
            f.write("\n")
            
            if stats.error_types:
                f.write("Error Type Distribution\n")
                f.write("---------------------\n")
                for error_type, count in stats.error_types.items():
                    f.write(f"{error_type}: {count} occurrences\n")

# async def get_decision_with_timeout(prompt_type: PromptType) -> tuple[Optional[Decision], TimeoutMetrics]:
# Update the function signature
async def get_decision_with_timeout(prompt_type: PromptType) -> tuple[Optional[Decision], TimeoutMetrics]:
    """
    Wrapper for get_decision that implements timeout handling with backoff
    
    Args:
        prompt_type: Type of prompt to use for decision
        
    Returns:
        Tuple of (Optional[Decision], TimeoutMetrics) where Decision is either 
        DecisionSystem1 or DecisionCot based on the prompt type
    """

    timeout_strategy = TimeoutStrategy()
    total_timeout_duration = 0
    
    while True:
        current_timeout = timeout_strategy.get_current_timeout()
        try:
            # Create task with timeout
            task = asyncio.create_task(get_decision(prompt_type))
            start_time = time.time()
            decision = await asyncio.wait_for(task, timeout=current_timeout)
            
            # If we get here, the call succeeded
            return decision, TimeoutMetrics(
                occurred=timeout_strategy.retry_count > 0,
                retry_count=timeout_strategy.retry_count,
                final_timeout_duration=current_timeout,
                total_timeout_duration=total_timeout_duration + (time.time() - start_time)
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            total_timeout_duration += duration
            logger.warning(
                f"API call timed out after {duration:.2f}s "
                f"(attempt #{timeout_strategy.retry_count + 1})"
            )
            
            if timeout_strategy.should_retry():
                timeout_strategy.increment_retry()
                continue
            else:
                # We've exceeded our retry limit
                return None, TimeoutMetrics(
                    occurred=True,
                    retry_count=timeout_strategy.retry_count,
                    final_timeout_duration=current_timeout,
                    total_timeout_duration=total_timeout_duration
                )

async def main():
    tracker = PerformanceTracker(PROMPT_TYPE.name)
    
    for attempt in range(MAX_CALLS_PER_PROMPT):
        start_time = time.time()
        try:
            decision, timeout_metrics = await get_decision_with_timeout(PROMPT_TYPE)
            execution_time = time.time() - start_time
            
            # Improved success checking
            if decision and hasattr(decision, 'prediction') and hasattr(decision, 'confidence'):
                # Validate the decision object more thoroughly
                if check_response_complete(PROMPT_TYPE, decision, OLLAMA_MODEL):
                    metrics = PromptPerformanceMetrics(
                        attempt_number=attempt + 1,
                        execution_time_seconds=execution_time,
                        successful=True,
                        timeout_metrics=timeout_metrics,
                        prediction=str(decision.prediction),
                        confidence=float(decision.confidence)
                    )
                else:
                    raise ValueError("Incomplete decision object")
            else:
                raise ValueError("Invalid decision object")
                
        except Exception as e:
            execution_time = time.time() - start_time
            metrics = PromptPerformanceMetrics(
                attempt_number=attempt + 1,
                execution_time_seconds=execution_time,
                successful=False,
                timeout_metrics=TimeoutMetrics(occurred=False),
                error_message=str(e)
            )
        
        # Ensure we always record the attempt
        tracker.record_attempt(metrics)

if __name__ == "__main__":
    asyncio.run(main())