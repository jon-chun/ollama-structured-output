# main.py

import asyncio
import logging
import time
import re
from typing import List, Optional, Set, Tuple, Dict
from collections import defaultdict
from pathlib import Path
import getpass

from config import load_config, Config
from data_manager import DataManager
from prompt_manager import PromptManager
from models import PromptType
from metrics import PromptMetrics, TimeoutMetrics
from performance import PerformanceTracker, PerformanceStats, save_aggregate_stats

# File Dependencies: config.yaml, util.py, decision.py,
#   data_manager.py, prompt_manager.py, models.py, metrics.py, performance.py

from utils import (
    check_and_pull_model,
    clean_model_name,
    get_prompt_type_str,
    get_completion_counts,
    is_combination_fully_complete,
    check_model_prompt_completion,
)

from decision import (
    get_decision_with_timeout,  # This will be overridden per API branch below.
    save_decision
)

# Global configuration settings for delays (read from config)
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
        else:
            logging.info(f"Model {model_name} is available.")
    return model_availability


async def cleanup_model():
    """Stop any running model instances without removing the model."""
    try:
        await asyncio.sleep(2)  # Allow resources to be freed
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

                if sample_calls[row_id] >= max_calls:
                    continue

                calls_needed = max_calls - sample_calls[row_id]
                for call_index in range(calls_needed):
                    try:
                        prompt = prompt_manager.get_prompt(prompt_type, row_id)
                        final_prompt = build_final_prompt(config, prompt)
                        start_time = time.time()
                        # Call the API using the branch‐specific decision function.
                        decision, meta_data, used_prompt, extra_data, timeout_metrics = await get_decision_with_timeout(
                            prompt_type=prompt_type,
                            model_name=model_name,
                            config=config,
                            prompt=final_prompt
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
                                repeat_index=sample_calls[row_id],
                                extra_data=extra_data
                            )

                            if save_success:
                                sample_calls[row_id] += 1
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
                        f"Completed sample {processed_samples}/{samples_needed} (ID: {row_id} with {max_calls} calls)"
                    )

            await asyncio.sleep(0.1)  # Prevent system overload

        cycle_duration = time.time() - cycle_start_time
        tracker.save_metrics(cycle_duration)
        logging.info(
            f"Evaluation cycle completed. Processed {len(unique_ids_processed)} samples. "
            f"Total unique samples with {max_calls} calls: {sum(1 for calls in sample_calls.values() if calls >= max_calls)}"
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
    output_base = Path(config.output["base_dir"])
    tracker = PerformanceTracker(
        prompt_type=prompt_type,
        model_name=model_name,
        output_base_dir=output_base
    )
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

        if len(tracker.metrics) > 0:
            tracker.save_metrics(session_duration)
            stats = tracker._generate_stats()
            if stats is not None:
                logging.info(
                    f"Generated statistics for {model_name} with {prompt_type} (Accuracy: {stats.prediction_accuracy:.2f}%, "
                    f"Attempts: {stats.total_attempts})"
                )
                return stats

        logging.info(f"No stats generated for {model_name} - {prompt_type} (no attempts recorded)")
        return None

    except Exception as e:
        logging.error(f"Critical error in evaluation session: {e}", exc_info=True)
        return None


async def process_model_evaluations(
    model_name: str,
    config: Config,
    data_manager: DataManager,
    prompt_manager: PromptManager,
    processed_combinations: Set[Tuple[str, str]],
    combination_completion_status: Dict[Tuple[str, str], Dict],
    output_base: Path,
    session_results: List[PerformanceStats]
):
    """
    Process evaluations for a given model across all prompt types.
    """
    logging.info(f"Starting evaluation of model: {model_name}")
    model_name_clean = clean_model_name(model_name)
    (output_base / model_name_clean).mkdir(parents=True, exist_ok=True)

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


async def main():
    """Main orchestrator with API-specific decision call integration."""
    config = load_config("config.yaml")
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
        processed_combinations: Set[Tuple[str, str]] = set()
        combination_completion_status: Dict[Tuple[str, str], Dict] = {}

        model_api_type = config.model_parameters['api_type'].lower()

        # ----- OPENAI API BRANCH -----
        if model_api_type == 'openai':
            logging.info("Using OpenAI model API")
            openai_key = getpass.getpass('Please enter your OPENAI_API_KEY: ')
            from openai import OpenAI
            # Initialize the client (assumes the client accepts an api_key parameter)
            openai_client = OpenAI(api_key=openai_key)
            # Define the structured tool for the decision call (adapted from your example)
            openai_tools = [{
                "type": "function",
                "function": {
                    "name": "get_decision",
                    "description": (
                        "Ignore any tags and text between <think> and </think> and parse from the bottom up to try to find "
                        "these 3 contiguous items in reverse order: 1. confidence_int - any integer after 'confidence:', "
                        "2. prediction_str - any capitalization variant of 'YES' or 'NO' after 'prediction:', "
                        "3. reason_str - a concatenated string of all reasoning given just before the 'prediction'"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prediction_str": {
                                "type": "string",
                                "description": 'A required word following "prediction": normalized to uppercase YES or NO',
                            },
                            "confidence_int": {
                                "type": "integer",
                                "description": 'An optional 0-100 integer following "confidence:" or -1 if not found',
                            },
                            "reason_str": {
                                "type": "string",
                                "description": 'A concatenated string of reasoning just before the "prediction", or empty if not found',
                            },
                        },
                        "required": ["prediction_str"],
                        "additionalProperties": False,
                    },
                    "strict": True
                }
            }]

            async def get_decision_with_timeout_openai(prompt_type: PromptType, model_name: str, config: Config, prompt: str):
                messages = [{"role": "user", "content": prompt}]
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",  # Replace with your desired model if needed.
                        messages=messages,
                        tools=openai_tools,
                    )
                    # Assume the API returns a structured function call in the following location:
                    tool_call = response.choices[0].message.tool_calls
                    # Create a dummy decision object (adjust as needed)
                    Decision = type("Decision", (), {"prediction": tool_call.get("prediction_str", ""),
                                                       "confidence": tool_call.get("confidence_int", -1)})
                    meta_data = {}
                    used_prompt = prompt
                    extra_data = {}
                    timeout_metrics = None
                    return Decision, meta_data, used_prompt, extra_data, timeout_metrics
                except Exception as e:
                    logging.error(f"OpenAI API call error: {e}")
                    return None, None, prompt, None, None

            # Override the common decision function with our OpenAI version.
            from decision import get_decision_with_timeout as _orig_get_decision_with_timeout
            get_decision_with_timeout = get_decision_with_timeout_openai

            # For OpenAI, we assume a single model specified in configuration.
            api_model = config.model_parameters.get('api_model')
            if not api_model:
                logging.error("OpenAI 'api_model' parameter not found in configuration.")
                return
            combo_key = (api_model, "OpenAI")
            completion_counts = get_completion_counts(output_base, api_model, "OpenAI")
            is_complete = is_combination_fully_complete(
                completion_counts,
                config.flags.max_samples,
                config.execution.max_calls_per_prompt
            )
            combination_completion_status[combo_key] = {'is_complete': is_complete, 'completion_counts': completion_counts}
            await process_model_evaluations(
                model_name=api_model,
                config=config,
                data_manager=data_manager,
                prompt_manager=prompt_manager,
                processed_combinations=processed_combinations,
                combination_completion_status=combination_completion_status,
                output_base=output_base,
                session_results=session_results
            )

        # ----- ANTHROPIC API BRANCH -----
        elif model_api_type == 'anthropic':
            logging.info("Using Anthropic model API")
            anthropic_key = getpass.getpass('Please enter your ANTHROPIC_API_KEY: ')
            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            # Define the structured tool for the decision call (adapted from your example)
            anthropic_tool = {
                "name": "get_decision",
                "description": (
                    "Ignore any tags and text between <think> and </think> and parse from the bottom up to try to find "
                    "these 3 contiguous items in reverse order: 1. confidence_int - any integer after 'confidence:', "
                    "2. prediction_str - any capitalization variant of 'YES' or 'NO' after 'prediction:', "
                    "3. reason_str - a concatenated string of all reasoning given just before the 'prediction'"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prediction_str": {
                            "type": "string",
                            "description": 'A required word following "prediction": normalized to uppercase YES or NO',
                        },
                        "confidence_int": {
                            "type": "integer",
                            "description": 'An optional 0-100 integer following "confidence:" or -1 if not found',
                        },
                        "reason_str": {
                            "type": "string",
                            "description": 'A concatenated string of reasoning just before the "prediction", or empty if not found',
                        },
                    },
                    "required": ["prediction_str"],
                },
            }

            async def get_decision_with_timeout_anthropic(prompt_type: PromptType, model_name: str, config: Config, prompt: str):
                messages = [{"role": "user", "content": prompt}]
                try:
                    response = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",  # Adjust model as needed.
                        max_tokens=1024,
                        tools=[anthropic_tool],
                        messages=messages,
                    )
                    # Assume response includes a structured result in a similar fashion.
                    tool_call = response.get("tool_calls", {})  # Adjust extraction per actual response.
                    Decision = type("Decision", (), {"prediction": tool_call.get("prediction_str", ""),
                                                       "confidence": tool_call.get("confidence_int", -1)})
                    meta_data = {}
                    used_prompt = prompt
                    extra_data = {}
                    timeout_metrics = None
                    return Decision, meta_data, used_prompt, extra_data, timeout_metrics
                except Exception as e:
                    logging.error(f"Anthropic API call error: {e}")
                    return None, None, prompt, None, None

            from decision import get_decision_with_timeout as _orig_get_decision_with_timeout
            get_decision_with_timeout = get_decision_with_timeout_anthropic

            api_model = config.model_parameters.get('api_model')
            if not api_model:
                logging.error("Anthropic 'api_model' parameter not found in configuration.")
                return
            combo_key = (api_model, "Anthropic")
            completion_counts = get_completion_counts(output_base, api_model, "Anthropic")
            is_complete = is_combination_fully_complete(
                completion_counts,
                config.flags.max_samples,
                config.execution.max_calls_per_prompt
            )
            combination_completion_status[combo_key] = {'is_complete': is_complete, 'completion_counts': completion_counts}
            await process_model_evaluations(
                model_name=api_model,
                config=config,
                data_manager=data_manager,
                prompt_manager=prompt_manager,
                processed_combinations=processed_combinations,
                combination_completion_status=combination_completion_status,
                output_base=output_base,
                session_results=session_results
            )

        # ----- GOOGLE API BRANCH -----
        elif model_api_type == 'google':
            from google import genai
            from google.genai import types

            logging.info("Using Google model API")
            google_key = getpass.getpass('Please enter your GOOGLE_API_KEY: ')

            google_client = genai.Client(api_key=google_key)
            MODEL_ID = config.model_parameters['api_model']
            # Define the structured function declaration per your example.
            get_destination = types.FunctionDeclaration(
                name="get_decision",
                description=(
                    "Ignore any tags and text between <think> and </think> and parse from the bottom up to try to find "
                    "these 3 contiguous items in reverse order (set value to empty if cannot be found): "
                    "1. confidence_int - any integer after 'confidence:', "
                    "2. prediction_str - any capitalization variant of 'YES' or 'NO' after 'prediction:', "
                    "3. reason_str - a concatenated string of all reasoning given just before the 'prediction'"
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prediction_str": {
                            "type": "STRING",
                            "description": 'A required word following "prediction": normalize to uppercase YES or NO',
                        },
                        "confidence_int": {
                            "type": "INTEGER",
                            "description": 'A optional 0-100 integer following "confidence:" or -1 if not found',
                        },
                        "reason_str": {
                            "type": "STRING",
                            "description": 'A concatenated string of reasoning just before the "prediction", or empty if not found',
                        },
                    },
                    "required": ["prediction_str"],
                },
            )
            destination_tool = types.Tool(
                function_declarations=[get_destination],
            )

            async def get_decision_with_timeout_google(prompt_type: PromptType, model_name: str, config: Config, prompt: str):
                try:
                    response = google_client.models.generate_content(
                        model=MODEL_ID,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            tools=[destination_tool],
                            temperature=0,
                        ),
                    )
                    # Extract the structured function call result.
                    tool_call = response.candidates[0].content.parts[0].function_call
                    Decision = type("Decision", (), {"prediction": tool_call.get("prediction_str", ""),
                                                       "confidence": tool_call.get("confidence_int", -1)})
                    meta_data = {}
                    used_prompt = prompt
                    extra_data = {}
                    timeout_metrics = None
                    return Decision, meta_data, used_prompt, extra_data, timeout_metrics
                except Exception as e:
                    logging.error(f"Google API call error: {e}")
                    return None, None, prompt, None, None

            from decision import get_decision_with_timeout as _orig_get_decision_with_timeout
            get_decision_with_timeout = get_decision_with_timeout_google

            api_model = config.model_parameters.get('api_model')
            if not api_model:
                logging.error("Google 'api_model' parameter not found in configuration.")
                return
            combo_key = (api_model, "gemini-1.5-flash-002','groq-1.5")
            completion_counts = get_completion_counts(output_base, api_model, "Google")
            is_complete = is_combination_fully_complete(
                completion_counts,
                config.flags.max_samples,
                config.execution.max_calls_per_prompt
            )
            combination_completion_status[combo_key] = {'is_complete': is_complete, 'completion_counts': completion_counts}
            await process_model_evaluations(
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
            logging.error(f"Unsupported model API type: {model_api_type}")
            return

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