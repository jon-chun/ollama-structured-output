# main.py

import asyncio
import logging
import time
from typing import List, Optional
from pathlib import Path

from config import load_config, Config
from data_manager import DataManager  # your code that loads data
from prompt_manager import PromptManager  # your code that provides get_prompt
from models import PromptType
from metrics import PromptMetrics, TimeoutMetrics
from performance import PerformanceTracker, PerformanceStats, save_aggregate_stats
from decision import (
    get_decision_with_timeout,
    save_decision
)

def build_final_prompt(config: Config, base_prompt: str) -> str:
    """
    Conditionally prepend and/or append prefix and suffix to the base prompt.
    """
    final_prompt = ""

    # Prepend prefix if FLAG_PROMPT_PREFIX is True
    if config.flags.FLAG_PROMPT_PREFIX:
        final_prompt += config.flags.prompt_prefix

    # Add the base prompt
    final_prompt += base_prompt

    # Append suffix if FLAG_PROMPT_SUFFIX is True
    if config.flags.FLAG_PROMPT_SUFFIX:
        final_prompt += config.flags.prompt_suffix

    return final_prompt

async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> None:
    """
    Run an evaluation cycle for a specific model & prompt type,
    respecting max_samples, batch_size and max_calls_per_prompt constraints.
    """
    # Get configuration constraints
    max_samples = config.flags.max_samples
    batch_size = config.batch_size
    max_calls = config.max_calls_per_prompt

    # Get dataset info and apply max_samples constraint
    dataset_info = data_manager.get_dataset_info()
    total_samples = min(
        dataset_info['dataset_sizes']['train'],
        max_samples if max_samples > 0 else float('inf')
    )

    logging.info(
        f"Starting evaluation cycle for {model_name} with {prompt_type}, "
        f"processing {total_samples} samples, batch size {batch_size}, "
        f"each repeated {max_calls} times"
    )

    processed_samples = 0
    
    while processed_samples < total_samples:
        # Calculate remaining samples and current batch size
        remaining_samples = total_samples - processed_samples
        current_batch_size = min(batch_size, remaining_samples)
        
        try:
            # Get batch of samples
            data_batch = data_manager.get_batch(current_batch_size, dataset='train')
            
            for sample in data_batch:
                if processed_samples >= total_samples:
                    break
                    
                row_id = sample['id']
                actual_value = sample['target']
                
                # Process each sample exactly max_calls times
                for repeat_index in range(max_calls):
                    start_time = time.time()
                    logging.info(
                        f"Processing sample {processed_samples + 1}/{total_samples} "
                        f"(ID: {row_id}), repeat #{repeat_index + 1}/{max_calls}"
                    )

                    try:
                        # Build the prompt
                        prompt = prompt_manager.get_prompt(prompt_type, row_id)
                        final_prompt = build_final_prompt(config, prompt)

                        # Get decision with timeout
                        decision, meta_data, used_prompt, extra_data, timeout_metrics = await get_decision_with_timeout(
                            prompt_type=prompt_type,
                            model_name=model_name,
                            config=config,
                            prompt=final_prompt
                        )

                        execution_time = time.time() - start_time

                        if decision is not None:
                            # Save valid decision
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

                            # Record metrics
                            attempt_num = (processed_samples + 1) * 1000 + repeat_index + 1
                            metrics = PromptMetrics(
                                attempt_number=attempt_num,
                                execution_time_seconds=execution_time,
                                successful=True,
                                timeout_metrics=timeout_metrics,
                                prediction=decision.prediction,
                                confidence=decision.confidence,
                                meta_data=meta_data.model_dump() if meta_data else {}
                            )
                        else:
                            # Record failed attempt
                            attempt_num = (processed_samples + 1) * 1000 + repeat_index + 1
                            metrics = PromptMetrics(
                                attempt_number=attempt_num,
                                execution_time_seconds=execution_time,
                                successful=False,
                                timeout_metrics=timeout_metrics,
                                error_message="No valid decision received"
                            )

                        tracker.record_attempt(metrics)

                    except Exception as e:
                        logging.error(f"Error processing sample {row_id} (repeat {repeat_index}): {str(e)}")
                
                processed_samples += 1
                if processed_samples >= total_samples:
                    break

        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            break

    logging.info(f"Evaluation cycle completed. Processed {processed_samples} samples.")



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
        tracker.save_metrics(session_duration)
        return tracker._generate_stats()
    except Exception as e:
        logging.error(f"Critical error in evaluation session: {e}")
        return None

async def main():
    """Main orchestrator for the evaluation process."""

    # Setup logging
    config = load_config("config.yaml")
    logging.basicConfig(
        level=getattr(logging, config.logging["level"].upper()),
        format=config.logging["format"],
        handlers=[
            logging.FileHandler(config.logging["file"]),
            logging.StreamHandler()
        ]
    )

    logging.debug(f"Loaded configuration: {config}")
    logging.debug(f"Max Samples: {config.max_samples}")
    logging.debug(f"Max Calls per Prompt: {config.max_calls_per_prompt}")
    logging.debug(f"Batch Size: {config.batch_size}")
    overall_start = time.time()


    try:
        # Prepare data
        data_manager = DataManager(config)
        data_manager.load_and_prepare_data()

        # Create output directory
        output_base = Path(config.output["base_dir"])
        output_base.mkdir(parents=True, exist_ok=True)

        # Prepare prompt manager
        prompt_manager = PromptManager(config, data_manager)

        session_results: List[PerformanceStats] = []

        # Evaluate each model
        for model_name, model_cfg in config.model_ensemble.items():
            logging.info(f"Starting evaluation of model: {model_name}")
            # Evaluate each prompt type
            for p_type in PromptType:
                logging.info(f"Testing prompt type: {p_type}")
                logging.info(f"Evaluation settings - Max Samples: {config.max_samples}, Max Calls per Prompt: {config.max_calls_per_prompt}")
                stats = await run_evaluation_session(
                    model_name=model_name,
                    prompt_type=p_type,
                    config=config,
                    data_manager=data_manager,
                    prompt_manager=prompt_manager
                )
                if stats:
                    session_results.append(stats)

        # Save aggregate
        total_duration = time.time() - overall_start
        if session_results:
            save_aggregate_stats(session_results, total_duration)
        else:
            logging.warning("No session results to aggregate.")

        logging.info(f"All evaluations completed in {total_duration:.2f} seconds.")

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
