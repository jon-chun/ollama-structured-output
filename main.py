# main.py
import asyncio
import logging
import time
from typing import List, Optional
from pathlib import Path

from config import load_config
from data_manager import DataManager  # your code that loads data
from prompt_manager import PromptManager  # your code that provides get_prompt
from models import PromptType
from metrics import PromptMetrics, TimeoutMetrics
from performance import PerformanceTracker, PerformanceStats, save_aggregate_stats
from decision import (
    get_decision_with_timeout,
    save_decision
)

async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config,
    tracker: PerformanceTracker,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> None:
    """
    Run evaluation for a model & prompt type on the training data,
    respecting the FLAG max_samples.
    """
    batch_size = config.execution.get("batch_size", 50)
    dataset_info = data_manager.get_dataset_info()

    total_train_samples = dataset_info['dataset_sizes']['train']
    # Use config.flags.max_samples or config.max_samples property
    if config.max_samples > 0:
        total_train_samples = min(total_train_samples, config.max_samples)

    logging.info(
        f"Starting evaluation cycle for {model_name} with {prompt_type} on up to {total_train_samples} samples."
    )

    remaining_samples = total_train_samples
    processed_samples = 0

    while remaining_samples > 0:
        current_batch_size = min(batch_size, remaining_samples)
        try:
            data_batch = data_manager.get_batch(current_batch_size)
            for sample in data_batch:
                start_time = time.time()
                row_id = sample['id']
                actual_value = sample['target']

                logging.info(
                    f"Processing sample {processed_samples + 1}/{total_train_samples} (ID: {row_id})"
                )

                try:
                    # Build prompt
                    prompt = prompt_manager.get_prompt(prompt_type, row_id)
                    # Call the model (with timeout)
                    decision, meta_data, used_prompt, timeout_metrics = await get_decision_with_timeout(
                        prompt_type=prompt_type,
                        model_name=model_name,
                        config=config,
                        prompt=prompt
                    )

                    execution_time = time.time() - start_time

                    if decision is not None:
                        # Save the decision
                        # We also store actual_value in meta_data for final performance tracking
                        meta_dict = {}
                        if meta_data:
                            meta_dict = meta_data.dict()
                        # embed the actual value for the PerformanceTracker
                        meta_dict['actual_value'] = actual_value

                        save_success = save_decision(
                            decision=decision,
                            meta_data=meta_data,
                            prompt_type=prompt_type,
                            model_name=model_name,
                            row_id=row_id,
                            actual_value=actual_value,
                            config=config,
                            used_prompt=used_prompt
                        )
                        if not save_success:
                            logging.warning("Decision valid but save failed")

                        # Record metrics
                        metrics = PromptMetrics(
                            attempt_number=processed_samples + 1,
                            execution_time_seconds=execution_time,
                            successful=True,
                            timeout_metrics=timeout_metrics,
                            prediction=decision.prediction,
                            confidence=decision.confidence,
                            meta_data=meta_dict
                        )
                    else:
                        # No decision returned
                        metrics = PromptMetrics(
                            attempt_number=processed_samples + 1,
                            execution_time_seconds=execution_time,
                            successful=False,
                            timeout_metrics=timeout_metrics,
                            error_message="No valid decision received"
                        )

                    tracker.record_attempt(metrics)
                except Exception as e:
                    logging.error(f"Error processing sample {row_id}: {str(e)}")
                processed_samples += 1

            remaining_samples -= current_batch_size

        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            break  # stop on batch error

async def run_evaluation_session(
    model_name: str,
    prompt_type: PromptType,
    config,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> Optional[PerformanceStats]:
    """Run a single model/prompt evaluation session."""
    from performance import PerformanceTracker
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
            from performance import save_aggregate_stats
            save_aggregate_stats(session_results, total_duration)
        else:
            logging.warning("No session results to aggregate.")

        logging.info(f"All evaluations completed in {total_duration:.2f} seconds.")

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
