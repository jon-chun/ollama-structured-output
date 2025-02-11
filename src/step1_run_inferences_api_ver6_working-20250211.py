import asyncio
import logging
import time
import re
from copy import deepcopy
import datetime
from typing import List, Optional, Set, Tuple, Dict
from collections import defaultdict
import json
from pathlib import Path
import getpass

from config import load_config, Config
from data_manager import DataManager
from prompt_manager import PromptManager
from models import PromptType
from metrics import PromptMetrics, TimeoutMetrics
from performance import PerformanceTracker, PerformanceStats, save_aggregate_stats

from dotenv import load_dotenv

# Load environment variables from .env (assumed to be in the same directory as decision.py)
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

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

# Global configuration settings for delays
config_import = load_config("config.yaml")
DELAY_BETWEEN_PROMPT_TYPES_SEC = config_import.timeout['delay_between_prompt_types_sec']
DELAY_BETWEEN_MODEL_LOAD_SEC = config_import.timeout['delay_between_model_load_sec']


async def check_model_availability(config: Config) -> Dict[str, bool]:
    # ... (unchanged) ...
    model_availability = {}
    for model_name in config.model_ensemble.keys():
        success, error = await check_and_pull_model(
            model_name,
            config.timeout.get('max_wait_ollama_pull_model_sec', 300)
        )
        model_availability[model_name] = success
        if not success:
            logging.error(f"Model {model_name} is not available and could not be pulled: {error}")
        else:
            logging.info(f"Model {model_name} is available.")
    return model_availability


async def cleanup_model():
    # ... (unchanged) ...
    try:
        await asyncio.sleep(2)
        import gc
        gc.collect()
        logging.info("Cleaned up model resources")
    except Exception as e:
        logging.warning(f"Error during model cleanup: {str(e)}")


def build_final_prompt(config: Config, base_prompt: str) -> str:
    # ... (unchanged) ...
    return (
        (config.flags.prompt_prefix if config.flags.FLAG_PROMPT_PREFIX else "") +
        base_prompt +
        (config.flags.prompt_suffix if config.flags.FLAG_PROMPT_SUFFIX else "")
    )


async def run_evaluation_cycle(
    api_type: str,
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker,
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
        completed_samples = sum(1 for calls in sample_calls.values() if calls >= max_calls)
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

            # Get samples excluding those with full call counts and already processed in this cycle
            completed_ids = {id for id, calls in sample_calls.items() if calls >= max_calls}
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
                        decision, meta_data, used_prompt, extra_data, timeout_metrics = await get_decision_with_timeout(
                            prompt_type=prompt_type,
                            api_type=api_type,
                            model_name=model_name,
                            config=config,
                            prompt=final_prompt
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
                                        'actual_value': actual_value,
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
    api_type: str,
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> Optional[PerformanceStats]:
    """Run a single model/prompt evaluation session."""
    output_base = Path(config.output["base_dir"])
    tracker = PerformanceTracker(
        prompt_type=prompt_type,
        model_name=model_name,
        output_base_dir=output_base
    )
    session_start = time.time()

    try:
        await run_evaluation_cycle(
            api_type,
            model_name,
            prompt_type,
            config,
            tracker,
            data_manager,
            prompt_manager
        )

        session_duration = time.time() - session_start

        if len(tracker.metrics) > 0:
            tracker.save_metrics(session_duration)
            stats = tracker._generate_stats()
            if stats is not None:
                logging.info(
                    f"Generated statistics for {model_name} with {prompt_type} "
                    f"(Accuracy: {stats.prediction_accuracy:.2f}%, Attempts: {stats.total_attempts})"
                )
                return stats

        logging.info(f"No stats generated for {model_name} - {prompt_type} (no attempts recorded)")
        return None

    except Exception as e:
        logging.error(f"Critical error in evaluation session: {e}", exc_info=True)
        return None
    

async def process_model_evaluations(
    api_type: str,
    model_name: str,
    config: Config,
    data_manager: DataManager,
    prompt_manager: PromptManager,
    processed_combinations: Set[Tuple[str, str]],
    combination_completion_status: Dict[Tuple[str, str], Dict],
    output_base: Path,
    session_results: List[PerformanceStats]
):
    logging.info(f"Starting evaluation of model: {model_name}")
    model_name_clean = clean_model_name(model_name)

    # Create model-specific output directory under the API-type folder.
    model_output_dir = output_base / model_name_clean
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for p_type in PromptType:
        combo_key = (model_name, str(p_type))
        if combo_key in processed_combinations:
            logging.info(f"Skipping already processed combination: {model_name} with {p_type}")
            continue

        if check_model_prompt_completion(
            output_base,
            model_name,
            str(p_type),
            config.flags.max_samples,
            config.execution.max_calls_per_prompt
        ):
            logging.info(f"Skipping {model_name} with {p_type}: maximum outputs reached")
            processed_combinations.add(combo_key)
            continue

        status = combination_completion_status.get(combo_key, {'is_complete': False})
        if status['is_complete']:
            logging.info(f"Skipping completed combination: {model_name} with {p_type}")
            processed_combinations.add(combo_key)
            continue

        logging.info(f"Processing combination: {model_name} with {p_type}")
        stats = await run_evaluation_session(
            api_type=api_type,
            model_name=model_name,
            prompt_type=p_type,
            config=config,
            data_manager=data_manager,
            prompt_manager=prompt_manager
        )
        if stats:
            session_results.append(stats)

        processed_combinations.add(combo_key)
        await cleanup_model()
        logging.info(f"Pausing for {DELAY_BETWEEN_PROMPT_TYPES_SEC} seconds between prompt types")
        await asyncio.sleep(DELAY_BETWEEN_PROMPT_TYPES_SEC)

    logging.info(f"Pausing for {DELAY_BETWEEN_MODEL_LOAD_SEC} seconds between models")
    await asyncio.sleep(DELAY_BETWEEN_MODEL_LOAD_SEC)


async def run_evaluation_for_config(config) -> None:
    logging.info("Starting evaluation process for current configuration")
    overall_start = time.time()

    session_results = []
    processed_combinations = set()
    combination_completion_status = {}

    # Setup data and prompts
    data_manager = DataManager(config)
    data_manager.load_and_prepare_data()

    # IMPORTANT: Update the output base directory to include the API type.
    # The common base is set in run_all_experiments; now append the API type.
    output_base = Path(config.output["base_dir"]) / config.model_parameters["api_type"]
    output_base.mkdir(parents=True, exist_ok=True)
    # Update the config so that subsequent functions use the correct base directory.
    config.output["base_dir"] = str(output_base)
    
    prompt_manager = PromptManager(config, data_manager)
    api_type = config.model_parameters["api_type"]

    # Then, process by API type as before.
    if api_type.lower() == 'openai':
        logging.info("Using OpenAI model API")
        api_model = config.model_parameters.get("api_model")
        if not api_model:
            logging.error("OpenAI api_model parameter not found in configuration.")
            return
        combo_key = (api_model, "OpenAI")
        completion_counts = get_completion_counts(output_base, api_model, "Openai")
        is_complete = is_combination_fully_complete(
            completion_counts,
            config.flags.max_samples,
            config.execution.max_calls_per_prompt
        )
        combination_completion_status[combo_key] = {
            'is_complete': is_complete,
            'completion_counts': completion_counts
        }
        await process_model_evaluations(
            api_type=api_type,
            model_name=api_model,
            config=config,
            data_manager=data_manager,
            prompt_manager=prompt_manager,
            processed_combinations=processed_combinations,
            combination_completion_status=combination_completion_status,
            output_base=output_base,
            session_results=session_results
        )
    elif api_type.lower() == 'anthropic':
        logging.info("Using Anthropic model API")
        api_model = config.model_parameters.get("api_model")
        if not api_model:
            logging.error("Anthropic api_model parameter not found in configuration.")
            return
        combo_key = (api_model, "Anthropic")
        completion_counts = get_completion_counts(output_base, api_model, "Anthropic")
        is_complete = is_combination_fully_complete(
            completion_counts,
            config.flags.max_samples,
            config.execution.max_calls_per_prompt
        )
        combination_completion_status[combo_key] = {
            'is_complete': is_complete,
            'completion_counts': completion_counts
        }
        await process_model_evaluations(
            api_type=api_type,
            model_name=api_model,
            config=config,
            data_manager=data_manager,
            prompt_manager=prompt_manager,
            processed_combinations=processed_combinations,
            combination_completion_status=combination_completion_status,
            output_base=output_base,
            session_results=session_results
        )

    elif api_type.lower() == 'google':
        from google import genai
        from google.genai import types
        logging.info("Using Google model API")
        model_name = config.model_parameters.get("api_model")
        if not model_name:
            logging.error("Google model_name parameter not found in configuration.")
            return
        for p_type in PromptType:
            combo_key = (model_name, str(p_type))
            if combo_key in processed_combinations:
                logging.info(f"Skipping duplicate model+prompt combination: {model_name} with {p_type}")
                continue
            completion_counts = get_completion_counts(output_base, model_name, str(p_type))
            is_complete = is_combination_fully_complete(
                completion_counts,
                config.flags.max_samples,
                config.execution.max_calls_per_prompt
            )
            combination_completion_status[combo_key] = {
                'is_complete': is_complete,
                'completion_counts': completion_counts
            }
        await process_model_evaluations(
            api_type=api_type,
            model_name=model_name,
            config=config,
            data_manager=data_manager,
            prompt_manager=prompt_manager,
            processed_combinations=processed_combinations,
            combination_completion_status=combination_completion_status,
            output_base=output_base,
            session_results=session_results
        )

    elif api_type.lower() == 'ollama':
        logging.info("Using Ollama model API")
        model_availability = await check_model_availability(config)
        for model_name, model_cfg in config.model_ensemble.items():
            if not model_availability.get(model_name, False):
                logging.error(f"Skipping model {model_name} - not available")
                continue
            for p_type in PromptType:
                combo_key = (model_name, str(p_type))
                if combo_key in processed_combinations:
                    logging.info(f"Skipping duplicate model+prompt combination: {model_name} with {p_type}")
                    continue
                completion_counts = get_completion_counts(output_base, model_name, str(p_type))
                is_complete = is_combination_fully_complete(
                    completion_counts,
                    config.flags.max_samples,
                    config.execution.max_calls_per_prompt
                )
                combination_completion_status[combo_key] = {
                    'is_complete': is_complete,
                    'completion_counts': completion_counts
                }
            await process_model_evaluations(
                api_type=api_type,
                model_name=model_name,
                config=config,
                data_manager=data_manager,
                prompt_manager=prompt_manager,
                processed_combinations=processed_combinations,
                combination_completion_status=combination_completion_status,
                output_base=output_base,
                session_results=session_results
            )

    elif api_type.lower() == 'groq':
        logging.info("Using Groq model API")
        api_model = config.model_parameters.get("api_model")
        if not api_model:
            logging.error("Groq api_model parameter not found in configuration.")
            return
        combo_key = (api_model, "Groq")
        completion_counts = get_completion_counts(output_base, api_model, "Groq")
        is_complete = is_combination_fully_complete(
            completion_counts,
            config.flags.max_samples,
            config.execution.max_calls_per_prompt
        )
        combination_completion_status[combo_key] = {
            'is_complete': is_complete,
            'completion_counts': completion_counts
        }
        await process_model_evaluations(
            api_type=api_type,
            model_name=api_model,
            config=config,
            data_manager=data_manager,
            prompt_manager=prompt_manager,
            processed_combinations=processed_combinations,
            combination_completion_status=combination_completion_status,
            output_base=output_base,
            session_results=session_results
        )

    elif api_type.lower() == 'together':
        logging.info("Using Together model API")
        api_model = config.model_parameters.get("api_model")
        if not api_model:
            logging.error("Together api_model parameter not found in configuration.")
            return
        combo_key = (api_model, "Together")
        completion_counts = get_completion_counts(output_base, api_model, "Together")
        is_complete = is_combination_fully_complete(
            completion_counts,
            config.flags.max_samples,
            config.execution.max_calls_per_prompt
        )
        combination_completion_status[combo_key] = {
            'is_complete': is_complete,
            'completion_counts': completion_counts
        }
        await process_model_evaluations(
            api_type=api_type,
            model_name=api_model,
            config=config,
            data_manager=data_manager,
            prompt_manager=prompt_manager,
            processed_combinations=processed_combinations,
            combination_completion_status=combination_completion_status,
            output_base=output_base,
            session_results=session_results
        )

    else:
        logging.error(f"Unsupported model API type: {api_type}")
        return
    
    total_duration = time.time() - overall_start
    if session_results:
        save_aggregate_stats(session_results, total_duration)
        logging.info("Successfully saved aggregate statistics")
    else:
        logging.warning("No session results to aggregate")
    logging.info(f"All evaluations completed in {total_duration:.2f} seconds")


async def run_all_experiments():
    """
    Iterates over all combinations of API types, models, seeds, temperatures, and versions
    in a round-robin fashion. For each unique combination of seed and temperature, the output
    base directory is common. Within that, subdirectories are created for each API type, then
    for each model, and then for each prompt type.
    """
    base_config = load_config("config.yaml")
    experiments = base_config.experiments

    # Access experiment parameters using dot-notation (Pydantic fields)
    api_types = experiments.api_types
    model_dict = experiments.model_dict
    seeds = experiments.seeds
    temperatures = experiments.temperatures
    versions = getattr(experiments, "versions", 1)

    # Determine the maximum number of models among all API types
    max_models = max(len(model_dict.get(api_type, [])) for api_type in api_types)

    # The base directory is defined solely by seed and temperature.
    for model_index in range(max_models):
        for api_type in api_types:
            models = model_dict.get(api_type, [])
            if model_index < len(models):
                model = models[model_index]
                # Iterate over versions (1 to versions)
                for version in range(1, versions + 1):
                    for seed in seeds:
                        for temp in temperatures:
                            config = deepcopy(base_config)
                            
                            # Set experiment parameters.
                            config.model_parameters["api_type"] = api_type
                            config.model_parameters["api_model"] = model  # Use the original model name for API calls.
                            config.model_parameters["model_temperature"] = temp
                            config.data["random_seed"] = seed
                            
                            # Set the common output directory (by seed and temperature).
                            config.output["base_dir"] = f"../evaluation_long_seed{seed}_temp{temp}"
                            # Log filename can include api_type, model, and version.
                            config.logging["file"] = f"evaluation_{api_type}_seed{seed}_temp{temp}_{model}_{version}.log"
                            
                            # --- Restartability Check ---
                            # Check in the API type subdirectory under the common base.
                            output_dir = Path(config.output["base_dir"]) / api_type
                            if output_dir.exists():
                                existing_counts = get_completion_counts(output_dir, config.model_parameters["api_model"], api_type)
                                if is_combination_fully_complete(existing_counts, config.flags.max_samples, config.execution.max_calls_per_prompt):
                                    logging.info(f"Skipping completed experiment: {config.output['base_dir']}/{api_type}")
                                    continue

                            logging.info(
                                f"Starting experiment: api_type={api_type}, model={model}_{version}, seed={seed}, temperature={temp}"
                            )
                            try:
                                await run_evaluation_for_config(config)
                            except Exception as e:
                                logging.error(
                                    f"Experiment failed for api_type={api_type}, model={model}_{version}, seed={seed}, temperature={temp}: {e}"
                                )
                            
                            await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    asyncio.run(run_all_experiments())
