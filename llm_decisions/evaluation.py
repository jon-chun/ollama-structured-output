import asyncio
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import time

from .config import Config, PromptType
from .models import (
    Decision, TimeoutMetrics, ApiCallMetadata,
    DecisionBase, DecisionCot
)
from .tracking import EnhancedPerformanceTracker
from .api_client import ModelClient
from .timeout import TimeoutStrategy

class ResponseProcessor:
    """Processes and validates model responses"""
    def __init__(self, prompt_type: PromptType):
        self.prompt_type = prompt_type
        self.decision_class = DecisionCot if prompt_type == PromptType.COT else DecisionBase

    async def process_response(self, response_content: str) -> Optional[Decision]:
        """Process and validate model response"""
        try:
            decision = self.decision_class.model_validate_json(response_content)
            await self._log_decision(decision)
            return decision
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            return None

    async def _log_decision(self, decision: Decision):
        """Log decision details"""
        logging.info(f"Prediction: {decision.prediction}")
        logging.info(f"Confidence: {decision.confidence}")
        
        if isinstance(decision, DecisionCot):
            for rf in decision.risk_factors:
                logging.info(f"Risk Factor: {rf.factor} ({rf.weight}): {rf.reasoning}")

async def get_decision(
    client: ModelClient,
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> Tuple[Optional[Decision], ApiCallMetadata]:
    """Get model decision with metrics"""
    processor = ResponseProcessor(prompt_type)
    start_time = time.time()
    
    try:
        response = await client.generate_response(
            model_name=model_name,
            prompt_type=prompt_type,
            config=config
        )
        
        decision = await processor.process_response(response.content)
        api_metadata = ApiCallMetadata.from_response(response, start_time)
        
        return decision, api_metadata
        
    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None, ApiCallMetadata.error_metadata(start_time)

async def get_decision_with_timeout(
    client: ModelClient,
    prompt_type: PromptType,
    model_name: str,
    config: Config
) -> Tuple[Optional[Decision], TimeoutMetrics, ApiCallMetadata]:
    """Get decision with timeout handling"""
    timeout_strategy = TimeoutStrategy(config)
    
    while True:
        attempt_start = time.time()
        current_timeout = timeout_strategy.get_current_timeout()
        
        try:
            decision, api_metadata = await asyncio.wait_for(
                get_decision(client, prompt_type, model_name, config),
                timeout=current_timeout
            )
            
            timeout_strategy.record_attempt(time.time() - attempt_start)
            return decision, timeout_strategy.get_metrics(), api_metadata
            
        except asyncio.TimeoutError:
            timeout_strategy.record_attempt(time.time() - attempt_start)
            
            if timeout_strategy.should_retry():
                timeout_strategy.increment_retry()
                continue
            else:
                return None, timeout_strategy.get_metrics(), ApiCallMetadata.timeout_metadata()
        
        except Exception as e:
            logging.error(f"Error during decision retrieval: {str(e)}")
            return None, timeout_strategy.get_metrics(), ApiCallMetadata.error_metadata(attempt_start)

async def run_evaluation_session(
    model_name: str,
    prompt_type: PromptType,
    config: Config
) -> Optional[PerformanceStats]:
    """Run complete evaluation session"""
    tracker = EnhancedPerformanceTracker(prompt_type, model_name)
    session_start = time.time()
    
    async with ModelClient(config) as client:
        for attempt in range(config.execution["max_calls_per_prompt"]):
            try:
                result = await get_decision_with_timeout(
                    client=client,
                    prompt_type=prompt_type,
                    model_name=model_name,
                    config=config
                )
                
                decision, timeout_metrics, api_metadata = result
                await tracker.record_attempt(
                    decision=decision,
                    timeout_metrics=timeout_metrics,
                    api_metadata=api_metadata,
                    attempt=attempt + 1
                )
                
            except Exception as e:
                logging.error(f"Error in evaluation attempt: {str(e)}")
                continue
    
    # Generate final statistics
    tracker.save_metrics(time.time() - session_start)
    return tracker.generate_stats()