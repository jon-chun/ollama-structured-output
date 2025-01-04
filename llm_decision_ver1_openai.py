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
from ollama import chat  # Make sure this import is correct in your environment

# -------------------
# 1) PROMPT TYPES
# -------------------
class PromptType(str, Enum):
    SYSTEM1 = 'system1'
    COT = 'cot'
    
# ... other classes remain unchanged ...

class Config:
    """Global configuration for managing application settings"""
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        config_path = Path("config.yaml")  # or your meta file
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._config["model_config"]

    @property
    def execution(self) -> Dict[str, Any]:
        return self._config["execution"]

    @property
    def timeout(self) -> Dict[str, Any]:
        return self._config["timeout"]

    @property
    def logging(self) -> Dict[str, Any]:
        return self._config["logging"]

    @property
    def output(self) -> Dict[str, Any]:
        return self._config["output"]

    @property
    def model_ensemble(self) -> Dict[str, Dict[str, Any]]:
        return self._config["model_ensemble"]

    # -------------------------------------------------------------
    # NEW: Access the prompt strings from config (system1, cot, etc.)
    # -------------------------------------------------------------
    @property
    def prompts(self) -> Dict[str, str]:
        return self._config["prompts"]

# ---------------------------
# 2) DATACLASS FOR METRICS
# ---------------------------
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
    
    # NEW: Time specifically spent on the Ollama API call
    api_call_duration: Optional[float] = None
    
    # NEW: Additional metadata from the Ollama response
    ollama_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.execution_time_seconds = float(self.execution_time_seconds)
        if self.prediction is not None:
            self.prediction = str(self.prediction)
        if self.confidence is not None:
            self.confidence = float(self.confidence)

# --------------------------------------------------------
# 3) get_decision() - Now capturing api_call_duration
#    and metadata from the response object
# --------------------------------------------------------
async def get_decision(
    prompt_type: PromptType, 
    model_name: str, 
    config: Config
) -> Tuple[Optional[Decision], Dict[str, Any], float]:
    """
    Get a decision from the model (without timeout handling).
    Returns: (DecisionObject or None, ollama_metadata dict, api_call_duration)
    """
    processor = ResponseProcessor(prompt_type)

    # 3.1) Retrieve the prompt from the config file
    prompt_str = config.prompts[prompt_type.value]

    try:
        # 3.2) Measure the API call's start and end times
        call_start = time.time()
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
        call_end = time.time()
        api_call_duration = call_end - call_start

        # 3.3) Try to extract metadata from the response object
        ollama_metadata = {}
        try:
            ollama_metadata["model"] = response.model
            ollama_metadata["created_at"] = response.created_at
            ollama_metadata["done_reason"] = response.done_reason
            ollama_metadata["done"] = response.done
            ollama_metadata["total_duration"] = response.total_duration
            ollama_metadata["load_duration"] = response.load_duration
            ollama_metadata["prompt_eval_count"] = response.prompt_eval_count
            ollama_metadata["prompt_eval_duration"] = response.prompt_eval_duration
            ollama_metadata["eval_count"] = response.eval_count
            ollama_metadata["eval_duration"] = response.eval_duration
        except AttributeError:
            pass  # if any field doesn't exist, ignore
        
        # 3.4) Validate the main content
        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            raise ValueError("Invalid API response structure (no message content)")

        decision = processor.process_response(response.message.content)
        if decision is None:
            raise ValueError("Failed to process response JSON")

        # Return the decision, the metadata, and the API duration
        return decision, ollama_metadata, api_call_duration

    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None, {}, 0.0

# ----------------------------------------------------------------------
# 4) get_decision_with_timeout() - integrate the new return signature
# ----------------------------------------------------------------------
async def get_decision_with_timeout(
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> Tuple[Optional[Decision], TimeoutMetrics, Dict[str, Any], float]:
    """
    Enhanced decision retrieval with sophisticated timeout handling.
    Returns:
      - Decision object if successful, else None
      - TimeoutMetrics
      - Ollama metadata dict
      - api_call_duration
    """
    timeout_strategy = TimeoutStrategy(config)

    while True:
        attempt_start = time.time()
        current_timeout = timeout_strategy.get_current_timeout()
        
        try:
            # Now we expect (decision, metadata, call_duration)
            decision, ollama_metadata, api_call_duration = await asyncio.wait_for(
                get_decision(prompt_type, model_name, config),
                timeout=current_timeout
            )
            attempt_duration = time.time() - attempt_start
            timeout_strategy.record_attempt(attempt_duration)

            return decision, timeout_strategy.get_metrics(), ollama_metadata, api_call_duration
            
        except asyncio.TimeoutError:
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
                # Return None for decision + final metrics
                return None, timeout_strategy.get_metrics(), {}, 0.0
        except Exception as e:
            attempt_duration = time.time() - attempt_start
            timeout_strategy.record_attempt(attempt_duration)
            logging.error(f"Error during decision retrieval: {str(e)}")
            return None, timeout_strategy.get_metrics(), {}, 0.0

# --------------------------------------------------------
# 5) run_evaluation_cycle() - record the extra fields
# --------------------------------------------------------
async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker
) -> None:
    for attempt in range(config.execution["max_calls_per_prompt"]):
        start_time = time.time()
        logging.info(
            f"Starting attempt #{attempt + 1} of {config.execution['max_calls_per_prompt']} "
            f"for model={model_name}, prompt_type={prompt_type}"
        )
        
        try:
            # We now get (decision, TimeoutMetrics, ollama_metadata, api_call_duration)
            decision, tmetrics, ollama_metadata, api_call_duration = await get_decision_with_timeout(
                prompt_type=prompt_type,
                model_name=model_name,
                config=config
            )
            execution_time = time.time() - start_time

            if decision is not None:
                logging.info(
                    f"Valid decision from {model_name}: "
                    f"{decision.prediction} ({decision.confidence}%)"
                )
                
                save_success = save_decision(decision, prompt_type, model_name, config)
                if not save_success:
                    logging.warning("Decision valid but save failed")
                
                # Record success
                metrics = PromptMetrics(
                    attempt_number=attempt + 1,
                    execution_time_seconds=execution_time,
                    successful=True,
                    timeout_metrics=tmetrics,
                    prediction=str(decision.prediction),
                    confidence=float(decision.confidence),
                    api_call_duration=api_call_duration,
                    ollama_metadata=ollama_metadata
                )
                
                # Print a short summary
                print(f"\nModel: {model_name}")
                print(f"Prompt Type: {prompt_type}")
                print(f"Prediction: {decision.prediction}")
                print(f"Confidence: {decision.confidence}%")
                if isinstance(decision, DecisionCot):
                    print("\nRisk Factor Analysis:")
                    for rf in decision.risk_factors:
                        print(f"- {rf.factor} ({rf.weight}): {rf.reasoning}")
            
            else:
                # Failed attempt
                error_msg = "No valid decision received"
                logging.warning(f"{error_msg} from {model_name}")
                metrics = PromptMetrics(
                    attempt_number=attempt + 1,
                    execution_time_seconds=execution_time,
                    successful=False,
                    timeout_metrics=tmetrics,
                    error_message=error_msg,
                    api_call_duration=api_call_duration,
                    ollama_metadata=ollama_metadata
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
                error_message=error_msg,
                api_call_duration=0.0,
                ollama_metadata=None
            )
        
        tracker.record_attempt(metrics)
        logging.info(
            f"Attempt #{attempt + 1} completed in {execution_time:.2f}s "
            f"(Status: {'Success' if metrics.successful else 'Failed'})"
        )

# ----------------------------------------------------
# 6) PerformanceTracker - Show new fields in the text
#    and JSON reports
# ----------------------------------------------------
class PerformanceTracker:
    """Tracks and analyzes performance metrics"""
    def __init__(self, prompt_type: str, model_name: str):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.metrics: List[PromptMetrics] = []
        self.start_time = datetime.now()

    def record_attempt(self, metrics: PromptMetrics):
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
            f"Execution time: {metrics.execution_time_seconds:.2f}s, "
            f"API call time: {metrics.api_call_duration:.2f}s{timeout_info}"
        )
        if not metrics.successful:
            logging.error(f"Error in attempt #{metrics.attempt_number}: {metrics.error_message}")

    # ... other methods remain the same ...

    def _save_text_report(self, stats: PerformanceStats, execution_time: float, timestamp: str):
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
            
            # ---------------------------------------
            # NEW: Log the API call duration + metadata
            # ---------------------------------------
            f.write("\nPer-Attempt Details:\n")
            for m in self.metrics:
                f.write(f"  Attempt #{m.attempt_number}:\n")
                f.write(f"    Execution Time: {m.execution_time_seconds:.2f}s\n")
                f.write(f"    API Call Time: {m.api_call_duration:.2f}s\n")
                if m.ollama_metadata:
                    f.write("    Ollama Metadata:\n")
                    for k, v in m.ollama_metadata.items():
                        f.write(f"      {k}: {v}\n")
                if not m.successful:
                    f.write(f"    ERROR: {m.error_message}\n")
                f.write("\n")
            
            # Timeout stats if present
            if stats.timeout_stats.total_timeouts > 0:
                f.write("\nTimeout Statistics\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Timeouts: {stats.timeout_stats.total_timeouts}\n")
                f.write(f"Average Duration: {stats.timeout_stats.avg_timeout_duration:.2f}s\n")
                f.write(f"Maximum Duration: {stats.timeout_stats.max_timeout_duration:.2f}s\n")
                f.write(f"Total Duration: {stats.timeout_stats.total_timeout_duration:.2f}s\n")

    def save_metrics(self, execution_time: float):
        """Save performance metrics to file (JSON + text)"""
        stats = self._generate_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            stats_dict = asdict(stats)
            
            # Also embed the per-attempt details including new fields
            # (api_call_duration, ollama_metadata)
            # You can attach them to stats_dict if desired or do a parallel structure.
            attempts_details = []
            for m in self.metrics:
                attempts_details.append(asdict(m))
            stats_dict["attempts"] = attempts_details
            
            json.dump(stats_dict, f, indent=2, default=str)
        
        # Save detailed text report
        self._save_text_report(stats, execution_time, timestamp)

# ----------------------------------------------
# The rest of your code (unchanged or minimal)
# ----------------------------------------------

async def main():
    # Everything else as before, but ensure the code references config.prompts
    # instead of PROMPT_SYSTEM1, PROMPT_COT from code.
    ...
