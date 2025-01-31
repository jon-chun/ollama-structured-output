# main.py

import asyncio
import logging
import time
import re
import datetime
from typing import List, Optional, Set, Tuple, Dict
from collections import defaultdict
import json
from pathlib import Path
from config import load_config, Config
from data_manager import DataManager
from prompt_manager import PromptManager
from models import PromptType
from metrics import PromptMetrics, TimeoutMetrics
from performance import PerformanceTracker, PerformanceStats, save_aggregate_stats

# File Dependencies: config.yaml, util.py, decision.py
#   data_manager.py, prompt_manager.py, models.py, metrics.py, performance.py   

from utils import (
    check_and_pull_model,
    clean_model_name,
    get_prompt_type_str,
    count_unique_samples,
    get_completion_status,
    get_next_sample,
    get_completion_counts,
    is_combination_fully_complete,
    check_model_prompt_completion,
)

from decision import (
    get_decision_with_timeout,  # Main inference function
    save_decision
)

# I/O subdirs/files and models are defined in config.yaml
config_import = load_config("config.yaml")

DELAY_BETWEEN_PROMPT_TYPES_SEC = config_import.timeout['delay_between_prompt_types_sec']
DELAY_BETWEEN_MODEL_LOAD_SEC = config_import.timeout['delay_between_model_load_sec']

async def check_model_availability(config: Config) -> Dict[str, bool]:
    """
    Check availability of all models in config and attempt to pull missing ones.
    Returns dict of model_name: available status.
    """
    model_availability = {}
    for model_name in config.model_ensemble.keys():
        success, error = await check_and_pull_model(
            model_name, 
            config.timeout.get('max_wait_ollama_pull_model_sec', 300)
        )
        model_availability[model_name] = success
        if not success:
            logging.error(f"Model {model_name} is not available and could not be pulled: {error}")
    return model_availability

async def cleanup_model():
    """Stop any running model instances without removing the model."""
    try:
        # Give a pause to allow resources to be freed
        await asyncio.sleep(2)
        
        # Force Python garbage collection
        import gc
        gc.collect()
        
        logging.info("Cleaned up model resources")
    except Exception as e:
        logging.warning(f"Error during model cleanup: {str(e)}")

def build_final_prompt(config: Config, base_prompt: str) -> str:
    """Conditionally prepend and/or append prefix and suffix to the base prompt."""
    return (
        (config.flags.prompt_prefix if config.flags.FLAG_PROMPT_PREFIX else "") +
        base_prompt +
        (config.flags.prompt_suffix if config.flags.FLAG_PROMPT_SUFFIX else "")
    )



async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker,  # We already receive the tracker!
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> None:
    """
    Run an evaluation cycle ensuring proper sample counts and repeat calls.
    Uses existing PerformanceTracker for statistics.
    """
    try:
        max_samples = config.flags.max_samples
        max_calls = config.execution.max_calls_per_prompt
        batch_size = config.batch_size
        
        prompt_type_str = get_prompt_type_str(prompt_type)
        output_dir = Path(config.output["base_dir"])
        
        # Track calls per sample using defaultdict
        sample_calls = defaultdict(int)
        
        # Count existing files per sample
        clean_name = clean_model_name(model_name)
        prompt_dir = output_dir / clean_name / prompt_type_str
        if prompt_dir.exists():
            for file_path in prompt_dir.glob(f"{model_name}_{prompt_type_str}_id*_*.json"):
                try:
                    match = re.search(r'_id(\d+)_', file_path.name)
                    if match:
                        sample_id = int(match.group(1))
                        sample_calls[sample_id] += 1
                except Exception as e:
                    logging.warning(f"Error parsing file {file_path}: {e}")

        # Calculate needed samples
        completed_samples = sum(1 for calls in sample_calls.values() 
                              if calls >= max_calls)
        samples_needed = max_samples - completed_samples
        
        if samples_needed <= 0:
            logging.info(
                f"All {max_samples} samples have {max_calls} calls each for "
                f"{model_name} with {prompt_type_str}"
            )
            return

        logging.info(
            f"Need {samples_needed} more samples with {max_calls} calls each for "
            f"{model_name} with {prompt_type_str}"
        )

        # Process samples
        processed_samples = 0
        unique_ids_processed = set()
        cycle_start_time = time.time()
        
        while processed_samples < samples_needed:
            current_batch_size = min(batch_size, samples_needed - processed_samples)
            
            # Get samples excluding those with full call counts
            completed_ids = {id for id, calls in sample_calls.items() 
                           if calls >= max_calls}
            data_batch = data_manager.get_batch(
                current_batch_size * 2,
                dataset='train',
                exclude_ids=completed_ids | unique_ids_processed
            )
            
            if not data_batch:
                logging.warning("No more available samples to process")
                break

            for sample in data_batch:
                if processed_samples >= samples_needed:
                    break
                    
                row_id = sample['id']
                actual_value = sample['target']
                
                # Skip if already fully processed
                if sample_calls[row_id] >= max_calls:
                    continue
                
                # Process this sample max_calls times
                calls_needed = max_calls - sample_calls[row_id]
                for call_index in range(calls_needed):
                    try:
                        # Generate prompt
                        prompt = prompt_manager.get_prompt(prompt_type, row_id)
                        final_prompt = build_final_prompt(config, prompt)
                        
                        # Make API call
                        start_time = time.time()
                        decision, meta_data, used_prompt, extra_data, timeout_metrics = (
                            await get_decision_with_timeout(
                                prompt_type=prompt_type,
                                model_name=model_name,
                                config=config,
                                prompt=final_prompt
                            )
                        )
                        execution_time = time.time() - start_time
                        
                        if decision is not None:
                            # Save the response
                            save_success = save_decision(
                                decision=decision,
                                meta_data=meta_data,
                                prompt_type=prompt_type,
                                model_name=model_name,
                                row_id=row_id,
                                actual_value=actual_value,
                                config=config,
                                used_prompt=used_prompt,
                                repeat_index=sample_calls[row_id],
                                extra_data=extra_data
                            )
                            
                            if save_success:
                                sample_calls[row_id] += 1
                                
                                # Record metrics using existing PerformanceTracker
                                metrics = PromptMetrics(
                                    attempt_number=processed_samples * max_calls + call_index + 1,
                                    execution_time_seconds=execution_time,
                                    successful=True,
                                    timeout_metrics=timeout_metrics,
                                    prediction=decision.prediction,
                                    confidence=decision.confidence,
                                    meta_data={
                                        'model': model_name,
                                        'prompt_type': str(prompt_type),
                                        'actual_value': actual_value,  # Critical for accuracy calculation
                                        **(meta_data.model_dump() if meta_data else {})
                                    }
                                )
                                tracker.record_attempt(metrics)
                                
                    except Exception as e:
                        logging.error(f"Error in call {call_index} for sample {row_id}: {e}")
                        continue
                
                if sample_calls[row_id] >= max_calls:
                    unique_ids_processed.add(row_id)
                    processed_samples += 1
                    logging.info(
                        f"Completed sample {processed_samples}/{samples_needed} "
                        f"(ID: {row_id} with {max_calls} calls)"
                    )
            
            await asyncio.sleep(0.1)  # Prevent overwhelming the system
        
        # Generate and save statistics using PerformanceTracker
        cycle_duration = time.time() - cycle_start_time
        tracker.save_metrics(cycle_duration)
        
        completed_count = len(unique_ids_processed)
        logging.info(
            f"Evaluation cycle completed. Processed {completed_count} samples. "
            f"Total unique samples with {max_calls} calls: "
            f"{sum(1 for calls in sample_calls.values() if calls >= max_calls)}"
        )
        
    except Exception as e:
        logging.error(f"Error in evaluation cycle: {str(e)}")
        raise


async def run_evaluation_session(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> Optional[PerformanceStats]:
    """Run a single model/prompt evaluation session."""
    # Create output base directory path
    output_base = Path(config.output["base_dir"])
    
    # Create tracker with output directory
    tracker = PerformanceTracker(
        prompt_type=prompt_type,
        model_name=model_name,
        output_base_dir=output_base
    )
    session_start = time.time()

    try:
        # Run the evaluation cycle
        await run_evaluation_cycle(
            model_name,
            prompt_type,
            config,
            tracker,
            data_manager,
            prompt_manager
        )
        
        # Calculate session duration
        session_duration = time.time() - session_start

        # Generate and save metrics if we have attempts
        if len(tracker.metrics) > 0:
            tracker.save_metrics(session_duration)
            stats = tracker._generate_stats()
            if stats is not None:
                logging.info(
                    f"Generated statistics for {model_name} with {prompt_type} "
                    f"(Accuracy: {stats.prediction_accuracy:.2f}%, "
                    f"Attempts: {stats.total_attempts})"
                )
                return stats
            
        logging.info(f"No stats generated for {model_name} - {prompt_type} (no attempts recorded)")
        return None

    except Exception as e:
        logging.error(f"Critical error in evaluation session: {e}", exc_info=True)
        return None
    

# In main.py, add this import at the top with other imports:
from utils import check_and_pull_model  # Add this import

# Add this new function before main():
async def check_model_availability(config: Config) -> Dict[str, bool]:
    """
    Check availability of all models in config and attempt to pull missing ones.
    Returns dict of model_name: available status.
    """
    model_availability = {}
    for model_name in config.model_ensemble.keys():
        success, error = await check_and_pull_model(
            model_name, 
            config.timeout.get('max_wait_ollama_pull_model_sec', 300)
        )
        model_availability[model_name] = success
        if not success:
            logging.error(f"Model {model_name} is not available and could not be pulled: {error}")
    return model_availability

# Then modify the main() function. Find the section that starts the model processing loop
# and update it to look like this:
async def main():
    """Main orchestrator with improved model management and restartability."""
    config = load_config("config.yaml")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging["level"].upper()),
        format=config.logging["format"],
        handlers=[
            logging.FileHandler(config.logging["file"]),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting evaluation process")
    overall_start = time.time()

    try:
        data_manager = DataManager(config)
        data_manager.load_and_prepare_data()

        output_base = Path(config.output["base_dir"])
        output_base.mkdir(parents=True, exist_ok=True)

        prompt_manager = PromptManager(config, data_manager)
        session_results: List[PerformanceStats] = []
        
        # Track processed combinations to avoid duplicates
        processed_combinations: Set[Tuple[str, str]] = set()
        
        # First, scan all existing output files
        combination_completion_status: Dict[Tuple[str, str], Dict] = {}
        
        # Model type
        model_api_type = config.model_parameters['api_type']

        if model_api_type == 'openai':
        
            # Process using OpenAI API calls with:
            # model = config.model_parameters['api_model']
            # temperature = config.model_parameters['model_temperature']
            # top_p = config.model_parameters['model_top_p']
            # max_tokens = config.model_parameters['model_max_tokens']
            pass

        elif model_api_type == 'anthropic':

            # Process using Claude API calls with:
            # model = config.model_parameters['api_model']
            # temperature = config.model_parameters['model_temperature']
            # top_p = config.model_parameters['model_top_p']
            # max_tokens = config.model_parameters['model_max_tokens']
            pass

        elif model_api_type == 'google':

            # Process using Google API calls with:
            # model = config.model_parameters['api_model']
            # temperature = config.model_parameters['model_temperature']
            # top_p = config.model_parameters['model_top_p']
            # max_tokens = config.model_parameters['model_max_tokens']
            pass

        elif model_api_type == 'ollama':
            logging.info("Using Ollama model API")

            # Check model availability before starting evaluations
            model_availability = await check_model_availability(config)
            
            # Pre-scan completion status for all combinations
            for model_name, model_cfg in config.model_ensemble.items():
                # Skip unavailable models
                if not model_availability[model_name]:
                    logging.error(f"Skipping model {model_name} - not available")
                    continue
                    
                for p_type in PromptType:
                    combo_key = (model_name, str(p_type))
                    
                    # Skip if we've already processed this combination
                    if combo_key in processed_combinations:
                        logging.info(f"Skipping duplicate model+prompt combination: {model_name} with {p_type}")
                        continue
                    
                    # Get completion counts for this combination
                    completion_counts = get_completion_counts(
                        output_base,
                        model_name,
                        str(p_type)
                    )
                    
                    is_complete = is_combination_fully_complete(
                        completion_counts,
                        config.flags.max_samples,
                        config.execution.max_calls_per_prompt
                    )
                    
                    combination_completion_status[combo_key] = {
                        'is_complete': is_complete,
                        'completion_counts': completion_counts
                    }

            # Now process only incomplete combinations for available models
            for model_name, model_cfg in config.model_ensemble.items():
                # Skip unavailable models
                if not model_availability[model_name]:
                    continue
                    
                logging.info(f"Starting evaluation of model: {model_name}")
                model_name_clean = clean_model_name(model_name)
                
                # Rest of the existing main() function continues as before...
                
                # Create model-specific output directory
                model_output_dir = output_base / model_name_clean
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Iterate over prompt types
                for p_type in PromptType:
                    combo_key = (model_name, str(p_type))
                    
                    # Skip if already processed or complete
                    if combo_key in processed_combinations:
                        logging.info(f"Skipping already processed combination: {model_name} with {p_type}")
                        continue


                    # Check if we've reached maximum outputs for this combination
                    if check_model_prompt_completion(
                        output_base, 
                        model_name, 
                        str(p_type),
                        config.flags.max_samples,
                        config.execution.max_calls_per_prompt
                    ):
                        logging.info(f"Skipping {model_name} with {p_type}: maximum outputs reached")
                        continue


                    status = combination_completion_status.get(combo_key, {
                        'is_complete': False,
                        'completion_counts': {}
                    })
                    
                    if status['is_complete']:
                        logging.info(f"Skipping completed combination: {model_name} with {p_type}")
                        continue
                    
                    logging.info(f"Processing combination: {model_name} with {p_type}")
                    
                    # Run evaluation
                    stats = await run_evaluation_session(
                        model_name=model_name,
                        prompt_type=p_type,
                        config=config,
                        data_manager=data_manager,
                        prompt_manager=prompt_manager
                    )
                    
                    if stats:
                        session_results.append(stats)
                    
                    # Mark as processed
                    processed_combinations.add(combo_key)
                    
                    # Cleanup and pause
                    await cleanup_model()
                    await asyncio.sleep(DELAY_BETWEEN_PROMPT_TYPES_SEC)
                
                # Longer pause between models
                await asyncio.sleep(DELAY_BETWEEN_MODEL_LOAD_SEC)

            # Save final results
            total_duration = time.time() - overall_start
            if session_results:
                save_aggregate_stats(session_results, total_duration)
                logging.info("Successfully saved aggregate statistics")
            else:
                logging.warning("No session results to aggregate")

            logging.info(f"All evaluations completed in {total_duration:.2f} seconds")

    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())