# main.py

import asyncio
import logging
import time
from typing import List, Optional, Set, Tuple, Dict
from pathlib import Path

from config import load_config, Config
from data_manager import DataManager
from prompt_manager import PromptManager
from models import PromptType
from metrics import PromptMetrics, TimeoutMetrics
from performance import PerformanceTracker, PerformanceStats, save_aggregate_stats
from utils import (
    clean_model_name,
    get_completion_status,
    get_next_sample,
    get_completion_counts,
    is_combination_fully_complete
)
from decision import (
    get_decision_with_timeout,
    save_decision
)

DELAY_BETWEEN_PROMPT_TYPES_SEC = 2
DELAY_BETWEEN_MODEL_LOAD_SEC = 10

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
    tracker: PerformanceTracker,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> None:
    """
    Run an evaluation cycle with configurable iterations and file checking.
    Now supports proper completion checking and restart capability.
    """
    max_samples = config.flags.max_samples
    max_calls = config.execution.max_calls_per_prompt
    batch_size = config.batch_size

    # Check overall completion status
    is_complete, existing_calls = get_completion_status(
        output_dir=Path(config.output["base_dir"]),
        model_name=model_name,
        prompt_type=str(prompt_type),
        max_samples=max_samples,
        max_calls=max_calls
    )

    if is_complete:
        logging.info(
            f"Skipping completed combination: {model_name} with {prompt_type} "
            f"(found {len(existing_calls)} samples with up to {max_calls} calls each)"
        )
        return

    # Calculate how many more samples we need
    completed_samples = sum(1 for calls in existing_calls.values() if calls >= max_calls)
    remaining_samples = max_samples - completed_samples

    if remaining_samples <= 0:
        logging.info(f"No additional samples needed for {model_name} with {prompt_type}")
        return

    dataset_info = data_manager.get_dataset_info()
    total_samples = min(
        dataset_info['dataset_sizes']['train'],
        remaining_samples
    )

    logging.info(
        f"Starting evaluation cycle for {model_name} with {prompt_type}, "
        f"processing {total_samples} more samples, batch size {batch_size}, "
        f"up to {max_calls} calls each"
    )

    processed_samples = 0
    
    while processed_samples < total_samples:
        remaining = total_samples - processed_samples
        current_batch_size = min(batch_size, remaining)
        
        try:
            data_batch = data_manager.get_batch(current_batch_size, dataset='train')
            
            for sample in data_batch:
                if processed_samples >= total_samples:
                    break
                    
                row_id = sample['id']
                actual_value = sample['target']
                
                # Get next needed repeat index for this sample
                next_repeat = get_next_sample(existing_calls, max_calls, row_id)
                if next_repeat is None:
                    continue  # Sample is complete
                
                # Process only remaining needed calls
                for repeat_index in range(next_repeat, max_calls):
                    start_time = time.time()
                    logging.info(
                        f"Processing sample {processed_samples + 1}/{total_samples} "
                        f"(ID: {row_id}), repeat #{repeat_index + 1}/{max_calls}"
                    )

                    try:
                        prompt = prompt_manager.get_prompt(prompt_type, row_id)
                        final_prompt = build_final_prompt(config, prompt)

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
                            save_success = save_decision(
                                decision=decision,
                                meta_data=meta_data,
                                prompt_type=prompt_type,
                                model_name=model_name,
                                row_id=row_id,
                                actual_value=actual_value,
                                config=config,
                                used_prompt=used_prompt,
                                repeat_index=repeat_index,
                                extra_data=extra_data
                            )
                            
                            if not save_success:
                                logging.warning("Decision valid but save failed")
                            else:
                                existing_calls[row_id] = existing_calls.get(row_id, 0) + 1

                            metrics = PromptMetrics(
                                attempt_number=(processed_samples + 1) * 1000 + repeat_index + 1,
                                execution_time_seconds=execution_time,
                                successful=True,
                                timeout_metrics=timeout_metrics,
                                prediction=decision.prediction,
                                confidence=decision.confidence,
                                meta_data=meta_data.model_dump() if meta_data else {}
                            )
                        else:
                            metrics = PromptMetrics(
                                attempt_number=(processed_samples + 1) * 1000 + repeat_index + 1,
                                execution_time_seconds=execution_time,
                                successful=False,
                                timeout_metrics=timeout_metrics,
                                error_message="No valid decision received"
                            )

                        tracker.record_attempt(metrics)

                    except Exception as e:
                        logging.error(f"Error processing sample {row_id} (repeat {repeat_index}): {str(e)}")
                        continue  # Continue to next repeat attempt
                
                processed_samples += 1
                if processed_samples >= total_samples:
                    break

        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            break

    completed_after = sum(1 for calls in existing_calls.values() if calls >= max_calls)
    logging.info(
        f"Evaluation cycle completed. Processed {processed_samples} samples. "
        f"Total completed samples: {completed_after}"
    )

async def run_evaluation_session(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> Optional[PerformanceStats]:
    """Run a single model/prompt evaluation session."""
    tracker = PerformanceTracker(prompt_type, model_name)
    session_start = time.time()

    try:
        await run_evaluation_cycle(
            model_name,
            prompt_type,
            config,
            tracker,
            data_manager,
            prompt_manager
        )
        session_duration = time.time() - session_start

        if len(tracker.attempts) > 0:  # Only save metrics if we have attempts
            tracker.save_metrics(session_duration)
            stats = tracker._generate_stats()
            if stats is not None:
                return stats
            
        logging.info(f"No stats generated for {model_name} - {prompt_type}")
        return None

    except Exception as e:
        logging.error(f"Critical error in evaluation session: {e}", exc_info=True)
        return None
    
def is_model_combination_complete(
    output_dir: Path,
    model_name: str,
    prompt_type: str,
    max_samples: int,
    max_calls: int
) -> bool:
    """Check if a model/prompt combination has completed all required samples."""
    clean_name = clean_model_name(model_name)
    model_dir = output_dir / clean_name
    
    if not model_dir.exists():
        return False
        
    # Track completed samples for this combination
    completed_samples = set()
    required_calls = {}
    
    # Check all output files for this combination
    for file_path in model_dir.glob(f"{model_name}_{prompt_type}_id*_*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                row_id = data['decision']['id']
                completed_samples.add(row_id)
                required_calls[row_id] = required_calls.get(row_id, 0) + 1
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Error reading file {file_path}: {e}")
            continue
    
    # Check if we have enough samples with enough calls each
    fully_completed = sum(1 for calls in required_calls.values() if calls >= max_calls)
    return fully_completed >= max_samples

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
        
        # Pre-scan completion status for all combinations
        for model_name, model_cfg in config.model_ensemble.items():
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

        # Now process only incomplete combinations
        for model_name, model_cfg in config.model_ensemble.items():
            logging.info(f"Starting evaluation of model: {model_name}")
            model_name_clean = clean_model_name(model_name)
            
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