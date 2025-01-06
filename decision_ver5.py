# decision.py
import json
import logging
from typing import Optional, Tuple, Dict, Any
import asyncio
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from ollama import chat

from config_ver6 import Config
from models import Decision, PromptType
from metrics import TimeoutMetrics
from util import pydantic_or_dict, convert_ns_to_s


class MetaData(BaseModel):
    """Model to represent metadata from an API response."""
    model: Optional[str] = Field(None, description="Model name")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    done_reason: Optional[str] = Field(None, description="Completion reason")
    done: Optional[bool] = Field(None, description="Completion status")
    total_duration: Optional[float] = Field(None, ge=0, description="Total duration in seconds")
    load_duration: Optional[float] = Field(None, ge=0, description="Model loading duration")
    prompt_eval_count: Optional[int] = Field(None, ge=0, description="Prompt token count")
    prompt_eval_duration: Optional[float] = Field(None, ge=0, description="Prompt evaluation duration")
    eval_count: Optional[int] = Field(None, ge=0, description="Response token count")
    eval_duration: Optional[float] = Field(None, ge=0, description="Response generation duration")


def extract_meta_data(response: Any) -> MetaData:
    """
    Safely extracts metadata fields from the API response,
    converting time measurements from nanoseconds to seconds where needed.
    """
    meta_data = {}

    meta_data["model"] = getattr(response, "model", None)
    meta_data["created_at"] = getattr(response, "created_at", None)
    meta_data["done_reason"] = getattr(response, "done_reason", None)
    meta_data["done"] = getattr(response, "done", None)

    # Convert timing fields from nanoseconds if needed
    timing_fields = [
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration"
    ]
    for field in timing_fields:
        value = getattr(response, field, None)
        if value is not None:
            meta_data[field] = float(value) / 1e9 if value > 1e7 else float(value)

    # Extract count fields
    for field in ["prompt_eval_count", "eval_count"]:
        meta_data[field] = getattr(response, field, None)

    return MetaData(**meta_data)


def process_model_response(response_text: str) -> Dict:
    """
    Attempt multiple parsing strategies to interpret the model output as JSON with
    {'prediction': 'YES'|'NO', 'confidence': 0..100}.
    """
    try:
        logging.debug(f"Raw model response: {response_text}")

        # Remove any markdown-style fences
        clean_text = response_text.replace('```json', '').replace('```', '')

        parsed_response = None

        # Strategy 1: direct JSON parse
        try:
            parsed_response = json.loads(clean_text.strip())
            logging.debug("Successfully parsed response as JSON")
        except json.JSONDecodeError:
            logging.debug("Failed to parse as direct JSON")

        # Strategy 2: find JSON content in text
        if not parsed_response:
            import re
            json_pattern = r'\{[^}]+\}'
            matches = re.findall(json_pattern, clean_text)
            for match in matches:
                try:
                    parsed_response = json.loads(match)
                    logging.debug("Extracted and parsed JSON content")
                    break
                except json.JSONDecodeError:
                    continue

        # Strategy 3: parse structured text for lines like "prediction: YES, confidence: 88"
        if not parsed_response:
            import re
            prediction_match = re.search(r'prediction[\s:"]*([YN]ES|NO)', clean_text, re.IGNORECASE)
            confidence_match = re.search(r'confidence[\s:"]*(\d+)', clean_text)
            if prediction_match and confidence_match:
                parsed_response = {
                    "prediction": prediction_match.group(1).upper(),
                    "confidence": int(confidence_match.group(1))
                }
                logging.debug("Parsed structured text response")

        if not parsed_response:
            raise ValueError("Could not extract valid response from model output")

        # Normalize
        normalized = {}

        pred_value = str(parsed_response.get('prediction', '')).upper()
        if pred_value in ['YES', 'NO']:
            normalized['prediction'] = pred_value
        else:
            # Fallback interpretation
            positive_indicators = ['YES', 'TRUE', '1', 'HIGH']
            if any(ind in pred_value for ind in positive_indicators):
                normalized['prediction'] = 'YES'
            else:
                normalized['prediction'] = 'NO'

        conf_value = parsed_response.get('confidence', None)
        if conf_value is not None:
            if isinstance(conf_value, str):
                conf_value = float(conf_value.replace('%', ''))
            normalized['confidence'] = int(min(max(float(conf_value), 0.0), 100.0))
        else:
            normalized['confidence'] = 90

        return normalized

    except Exception as e:
        logging.error(f"Error processing model response: {str(e)}")
        logging.error(f"Raw response: {response_text}")
        raise


async def get_decision(
    prompt_type: PromptType,
    model_name: str,
    config: Config,
    prompt: str
) -> Tuple[Optional[Decision], Optional[MetaData], str]:
    """
    Returns a tuple of (Decision, MetaData, used_prompt).
    """
    try:
        system_message = (
            "You are a risk assessment expert. Your responses must be in valid JSON format "
            "containing exactly two fields: 'prediction' and 'confidence'."
        )

        response = await asyncio.to_thread(
            chat,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            model=model_name,
            options={
                'temperature': config.model_parameters["model_temperature"],
                'top_p': config.model_parameters["model_top_p"],
                'max_tokens': config.model_parameters["model_max_tokens"],
            }
        )

        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            raise ValueError("Invalid API response structure")

        # Process
        normalized_response = process_model_response(response.message.content)

        # Construct a Decision
        try:
            decision = Decision(
                prediction=normalized_response['prediction'],
                confidence=normalized_response['confidence']
            )
        except ValidationError as ve:
            logging.error(f"Validation error creating Decision: {str(ve)}")
            logging.error(f"Normalized response: {normalized_response}")
            return None, None, prompt

        # Extract metadata
        meta_data = MetaData(
            model=getattr(response, 'model', None),
            created_at=getattr(response, 'created_at', None),
            done_reason=getattr(response, 'done_reason', None),
            done=getattr(response, 'done', None),
            total_duration=getattr(response, 'total_duration', None),
            load_duration=getattr(response, 'load_duration', None),
            prompt_eval_count=getattr(response, 'prompt_eval_count', None),
            prompt_eval_duration=getattr(response, 'prompt_eval_duration', None),
            eval_count=getattr(response, 'eval_count', None),
            eval_duration=getattr(response, 'eval_duration', None)
        )

        return decision, meta_data, prompt

    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None, None, prompt


async def get_decision_with_timeout(
    prompt_type: PromptType,
    model_name: str,
    config: Config,
    prompt: str
) -> Tuple[Optional[Decision], Optional[MetaData], str, TimeoutMetrics]:
    """
    Wraps get_decision(...) in an asyncio timeout.
    Returns (Decision, MetaData, used_prompt, TimeoutMetrics).
    """
    timeout_metrics = TimeoutMetrics(
        occurred=False,
        retry_count=0
    )
    timeout_seconds = float(config.model_parameters.get("api_timeout", 30.0))

    try:
        decision, meta_data, used_prompt = await asyncio.wait_for(
            get_decision(prompt_type, model_name, config, prompt),
            timeout=timeout_seconds
        )
        return decision, meta_data, used_prompt, timeout_metrics

    except asyncio.TimeoutError:
        logging.warning("API call timed out while waiting for model response...")
        timeout_metrics.occurred = True
        return None, None, prompt, timeout_metrics


def save_decision(
    decision: Decision,
    meta_data: MetaData,
    prompt_type: PromptType,
    model_name: str,
    row_id: int,
    actual_value: str,
    config: Config,
    used_prompt: str
) -> bool:
    """
    Save decision + metadata + used_prompt to a JSON file in the model's directory.
    """
    try:
        output_dir = Path(config.output["base_dir"]) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{model_name}_{prompt_type}_id{row_id}_{timestamp}.json"

        decision_data = pydantic_or_dict(decision)
        meta_data_data = pydantic_or_dict(meta_data)

        # Mark correctness
        decision_data.update({
            'id': row_id,
            'actual': actual_value,
            'correct': "YES" if str(decision.prediction).upper() == actual_value.upper() else "NO"
        })

        meta_data_data = convert_ns_to_s(meta_data_data)

        combined_data = {
            "decision": decision_data,
            "meta_data": meta_data_data,
            "evaluation": {
                "timestamp": timestamp,
                "model": model_name,
                "prompt_type": str(prompt_type),
                "row_id": row_id,
                "prediction_matches_actual": decision_data['correct']
            },
            "prompt": used_prompt
        }

        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, default=str)

        logging.info(f"Successfully saved decision+meta_data to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {str(e)}")
        return False
