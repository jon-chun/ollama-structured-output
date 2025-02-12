# decision.py

import json
import logging
from typing import Optional, Tuple, Dict, Any
import asyncio
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from ollama import chat

from config import Config
from models import Decision, PromptType
from metrics import TimeoutMetrics
from utils import clean_model_name, pydantic_or_dict, convert_ns_to_s, check_existing_decision

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

def generate_output_path(
    model_name: str,
    prompt_type: PromptType,
    row_id: int,
    repeat_index: int,
    timestamp: str,
    config: Config,
    output_dir: Path,
    nshot_ct: Optional[int] = None
) -> Path:
    """Generate the output path for a decision file."""
    if prompt_type == PromptType.COT_NSHOT:
        filename = (
            f"{model_name}_{prompt_type}_id{row_id}_"
            f"nshot{nshot_ct}_{timestamp}.json"
        )
    else:
        filename = (
            f"{model_name}_{prompt_type}_id{row_id}_"
            f"ver{repeat_index}_{timestamp}.json"
        )
    return output_dir / filename

def process_model_response(response_text: str, model_name: str) -> Dict:
    """
    Attempt multiple parsing strategies to interpret the model output as JSON
    with {'prediction': 'YES'|'NO', 'confidence': 0..100, 'risk_factors': [...] }.
    """
    try:
        logging.debug(f"Raw model response: {response_text}")
        clean_text = response_text.replace('```json', '').replace('```', '')
        parsed_response = None

        # Strategy 1: Direct JSON parsing
        try:
            parsed_response = json.loads(clean_text.strip())
            logging.debug("Successfully parsed response as JSON")
        except json.JSONDecodeError:
            logging.debug("Failed to parse as direct JSON")

        # Strategy 2: Extract JSON from text using regex
        if not parsed_response:
            import re
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, clean_text, re.DOTALL)
            for match in matches:
                try:
                    parsed_response = json.loads(match)
                    logging.debug("Extracted and parsed JSON content using regex")
                    break
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Structured text parsing for specific fields
        if not parsed_response:
            import re
            prediction_match = re.search(r'prediction[\s:"]*([YN]ES|NO)', clean_text, re.IGNORECASE)
            confidence_match = re.search(r'confidence[\s:"]*(\d+)', clean_text)
            risk_factors_match = re.search(r'risk_factors[\s:"]*(\[.*\])', clean_text, re.IGNORECASE)
            if prediction_match and confidence_match:
                parsed_response = {
                    "prediction": prediction_match.group(1).upper(),
                    "confidence": int(confidence_match.group(1))
                }
                if risk_factors_match:
                    parsed_response["risk_factors"] = json.loads(risk_factors_match.group(1))
                logging.debug("Parsed structured text response")
        
        if not parsed_response:
            raise ValueError("Could not extract valid response from model output")

        # Normalize the response
        normalized = {}
        pred_value = str(parsed_response.get('prediction', '')).upper()
        normalized['prediction'] = (
            pred_value if pred_value in ['YES', 'NO']
            else 'YES' if any(ind in pred_value for ind in ['YES', 'TRUE', '1', 'HIGH'])
            else 'NO'
        )

        # Handle confidence
        conf_value = parsed_response.get('confidence', None)
        if conf_value is not None:
            if isinstance(conf_value, str):
                conf_value = float(conf_value.replace('%', ''))
            normalized['confidence'] = int(min(max(float(conf_value), 0.0), 100.0))
        else:
            normalized['confidence'] = 90  # Default confidence

        # Handle risk_factors
        normalized['risk_factors'] = parsed_response.get('risk_factors', None)

        logging.debug(f"Normalized response: {normalized}")
        return normalized

    except Exception as e:
        logging.error(f"Error processing model: {str(model_name)}")
        logging.error(f"Error processing model response: {str(e)}")
        logging.error(f"Raw response: {response_text}")
        raise

async def get_decision(
    prompt_type: PromptType,
    model_name: str,
    config: Config,
    prompt: str
) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    """Returns a tuple of (Decision, MetaData, used_prompt, extra_data)."""
    try:
        system_message = config.prompts.get("prompt_persona", "") or (
            "You are a risk assessment expert. Your responses must be in valid JSON format "
            "containing exactly three fields: 'risk_factors', 'prediction', and 'confidence'."
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

        normalized_response = process_model_response(response.message.content, model_name)

        try:
            decision = Decision(
                prediction=normalized_response['prediction'],
                confidence=normalized_response['confidence']
            )
        except ValidationError as ve:
            logging.error(f"Validation error creating Decision: {str(ve)}")
            logging.error(f"Normalized response: {normalized_response}")
            return None, None, prompt, {}

        extra_data = {}
        if 'risk_factors' in normalized_response and normalized_response['risk_factors']:
            extra_data['risk_factors'] = normalized_response['risk_factors']

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

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None, None, prompt, {}

async def get_decision_with_timeout(
    prompt_type: PromptType,
    model_name: str,
    config: Config,
    prompt: str
) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any], TimeoutMetrics]:
    """Wraps get_decision with timeout handling."""
    timeout_metrics = TimeoutMetrics(
        occurred=False,
        retry_count=0,
        total_timeout_duration=0.0
    )
    # timeout_seconds = float(config.model_parameters.get("api_timeout", 30.0))
    timeout_seconds = float(config.model_ensemble[model_name]["max_response_time"])

    try:
        logging.debug(f"Calling get_decision with prompt: {prompt[:50]}...")
        decision, meta_data, used_prompt, extra_data = await asyncio.wait_for(
            get_decision(prompt_type, model_name, config, prompt),
            timeout=timeout_seconds
        )
        logging.debug("Received response from get_decision.")
        return decision, meta_data, used_prompt, extra_data, timeout_metrics

    except asyncio.TimeoutError:
        logging.warning("API call timed out while waiting for model response...")
        timeout_metrics.occurred = True
        timeout_metrics.retry_count += 1
        return None, None, prompt, {}, timeout_metrics

    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None, None, prompt, {}, timeout_metrics

def save_decision(
    decision: Decision,
    meta_data: MetaData,
    prompt_type: PromptType,
    model_name: str,
    row_id: int,
    actual_value: str,
    config: Config,
    used_prompt: str,
    repeat_index: int = 0,
    extra_data: Dict[str, Any] = None
) -> bool:
    """Save decision and metadata to filesystem."""
    if extra_data is None:
        extra_data = {}

    try:
        clean_name = clean_model_name(model_name)
        output_dir = Path(config.output["base_dir"]) / clean_name
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = generate_output_path(
            model_name=model_name,
            prompt_type=prompt_type,
            row_id=row_id,
            repeat_index=repeat_index,
            timestamp=timestamp,
            config=config,
            output_dir=output_dir,
            nshot_ct=config.execution.nshot_ct if prompt_type == PromptType.COT_NSHOT else None
        )

        decision_data = pydantic_or_dict(decision)
        meta_data_data = pydantic_or_dict(meta_data)

        decision_data.update({
            'id': row_id,
            'actual': actual_value,
            'correct': "YES" if decision.prediction.upper() == actual_value.upper() else "NO"
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
                "prediction_matches_actual": decision_data['correct'],
                "repeat_index": repeat_index
            }
        }

        if prompt_type == PromptType.COT_NSHOT:
            combined_data["evaluation"]["nshot_ct"] = config.execution.nshot_ct

        if 'risk_factors' in extra_data:
            combined_data["risk_factors"] = extra_data["risk_factors"]
        
        combined_data["prompt"] = used_prompt

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, default=str)

        logging.info(f"Successfully saved decision+meta_data to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {str(e)}")
        return False