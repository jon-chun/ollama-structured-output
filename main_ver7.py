# main.py

import asyncio
import logging
import time
from typing import List, Optional
from pathlib import Path

from config_ver7 import load_config, Config
from data_manager_ver7 import DataManager  # your code that loads data
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
    if config.flags.get("FLAG_PROMPT_PREFIX", False):
        prefix_text = config.flags.get("prompt_prefix", "")
        final_prompt += prefix_text

    # Add the base prompt
    final_prompt += base_prompt

    # Append suffix if FLAG_PROMPT_SUFFIX is True
    if config.flags.get("FLAG_PROMPT_SUFFIX", False):
        suffix_text = config.flags.get("prompt_suffix", "")
        final_prompt += suffix_text

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
    now with repeated calls for each sample.
    """
    # Get batch size and the max calls per sample
    batch_size = config.batch_size
    max_calls = config.max_calls_per_prompt

    dataset_info = data_manager.get_dataset_info()
    total_train_samples = dataset_info['dataset_sizes']['train']
    # Limit total samples if config.flags.max_samples > 0
    if config.max_samples > 0:
        total_train_samples = min(total_train_samples, config.flags["max_samples"])

    logging.info(
        f"Starting evaluation cycle for {model_name} with {prompt_type}, "
        f"up to {total_train_samples} samples, each repeated {max_calls} times."
    )

    remaining_samples = total_train_samples
    processed_samples = 0

    while remaining_samples > 0:
        current_batch_size = min(batch_size, remaining_samples)
        try:
            data_batch = data_manager.get_batch(current_batch_size, dataset='train')
            for sample in data_batch:
                row_id = sample['id']
                actual_value = sample['target']

                # -- NEW: We'll call the same prompt multiple times --
                for repeat_index in range(max_calls):
                    start_time = time.time()
                    logging.info(
                        f"Processing sample {processed_samples + 1}/"
                        f"{total_train_samples} (ID: {row_id}), repeat #{repeat_index+1}"
                    )

                    try:
                        # Build the prompt from row_id
                        prompt = prompt_manager.get_prompt(prompt_type, row_id)
                        # Apply prefix/suffix if enabled
                        final_prompt = build_final_prompt(config, prompt)

                        # Call the model (with timeout)
                        decision, meta_data, used_prompt, extra_data, timeout_metrics = await get_decision_with_timeout(
                            prompt_type=prompt_type,
                            model_name=model_name,
                            config=config,
                            prompt=final_prompt
                        )

                        execution_time = time.time() - start_time

                        if decision is not None:
                            # Save each repeated call with a version index
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
                                extra_data=extra_data,  # pass the risk_factors
                            )
                            if not save_success:
                                logging.warning("Decision valid but save failed")

                            # Record the repeated call in PerformanceTracker
                            # We treat each repeated call as a separate attempt
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
                            # No decision received
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

                # After we've repeated calls for this sample, increment processed_samples
                processed_samples += 1

        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            # Depending on your preference, continue or break
            break

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
    config = load_config("config.yaml")

    logging.debug(f"Configuration Loaded: {config}")
    logging.debug(f"Max Samples: {config.max_samples}")
    logging.debug(f"Max Calls per Prompt: {config.max_calls_per_prompt}")
    logging.debug(f"Batch Size: {config.batch_size}")

    overall_start = time.time()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging["level"].upper()),
        format=config.logging["format"],
        handlers=[
            logging.FileHandler(config.logging["file"]),
            logging.StreamHandler()
        ]
    )

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
        logging.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
