# decision.py
import os
from dotenv import load_dotenv # find_dotenv
from pathlib import Path
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any
import asyncio
from datetime import datetime
from pathlib import Path
import getpass
from pydantic import BaseModel, Field, ValidationError
from ollama import chat
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from config import Config
from models import Decision, PromptType
from metrics import TimeoutMetrics
from utils import clean_model_name, pydantic_or_dict, convert_ns_to_s, check_existing_decision
# Make sure you have installed json-repair:
#   pip install json-repair
import json_repair

# Load the environment variables from the .env file
# Construct path relative to the current script's location
dotenv_path = Path(__file__).parent / "src" / ".env"  # __file__ is the current file's path
load_dotenv(dotenv_path)
# load_dotenv()

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
    """
    Generate the output path for a decision file.
    Maps PromptType enum to config.yaml prompt keys for directory structure.
    
    Args:
        model_name: Name of the model being evaluated
        prompt_type: Type of prompt being used (PromptType enum)
        row_id: ID of the current data row
        repeat_index: Index of current repeat attempt
        timestamp: Current timestamp string
        config: Configuration object
        output_dir: Base output directory
        nshot_ct: Number of shots for n-shot learning (optional)
        
    Returns:
        Path object representing the complete file path
    """
    # Get the correct prompt type string using our utility function
    from utils import get_prompt_type_str
    prompt_type_str = get_prompt_type_str(prompt_type)
    
    # Generate the filename based on prompt type
    if prompt_type == PromptType.COT_NSHOT:
        filename = (
            f"{model_name}_{prompt_type_str}_id{row_id}_"
            f"nshot{nshot_ct}_{timestamp}.json"
        )
    else:
        filename = (
            f"{model_name}_{prompt_type_str}_id{row_id}_"
            f"ver{repeat_index}_{timestamp}.json"
        )
    
    # Create the nested directory structure
    model_dir = output_dir / prompt_type_str
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Return the complete path
    return model_dir / filename
def check_existing_output(output_dir: Path, model_name: str, prompt_type: PromptType, 
                         row_id: int) -> bool:
    """
    Check if output file already exists for this model-prompt-id combination.
    Uses the correct nested directory structure based on prompt type.
    Returns True if exists, False otherwise.
    """
    # Map PromptType enum to config.yaml prompt keys
    prompt_type_to_key = {
        PromptType.SYSTEM1: 'system1',
        PromptType.COT: 'cot',
        PromptType.COT_NSHOT: 'cot-nshot'
    }
    
    prompt_type_str = prompt_type_to_key[prompt_type]
    clean_name = clean_model_name(model_name)
    model_dir = output_dir / clean_name / prompt_type_str
    
    if not model_dir.exists():
        return False
    
    # Check for any matching files ignoring ver{n} and datetime parts
    pattern = f"{model_name}_{prompt_type_str}_id{row_id}_*.json"
    matching_files = list(model_dir.glob(pattern))
    
    return len(matching_files) > 0


def remove_think_tags_and_triple_quotes(text: str) -> str:
    """
    1) Remove <THINK>...</THINK> tags and all enclosed text.
    2) Remove any triple quotes .
    Returns a cleaned-up string.
    """
    import re
    # Remove <THINK> ... </THINK>
    text = re.sub(r'<THINK>.*?</THINK>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove triple quotes of all forms
    triple_quotes_patterns = [r'```', r'"""', r"'''"]
    for pattern in triple_quotes_patterns:
        text = re.sub(pattern, '', text)

    return text.strip()

def process_model_response(response_text: str, model_name: str) -> Dict:
    """
    Attempt multiple parsing strategies to interpret the model output as JSON
    with {'prediction': 'YES'|'NO', 'confidence': 0..100, 'risk_factors': [...] }.
    """
    try:
        logging.debug(f"Raw model response: {response_text}")
        # Remove ```json or ``` from the original text for cleanliness
        clean_text = response_text.replace('```json', '').replace('```', '')
        parsed_response = None

        #
        # --- Strategy 1: Direct JSON parsing ---
        #
        try:
            parsed_response = json.loads(clean_text.strip())
            logging.debug("Successfully parsed response as JSON (Strategy 1)")
            # Keep original text
            parsed_response['original_text'] = response_text
        except json.JSONDecodeError:
            logging.debug("Failed to parse as direct JSON (Strategy 1)")

        #
        # --- Strategy 2: Extract JSON from text using regex ---
        #
        if not parsed_response:
            import re
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, clean_text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    parsed['original_text'] = response_text
                    parsed_response = parsed
                    logging.debug("Extracted and parsed JSON content using regex (Strategy 2)")
                    break
                except json.JSONDecodeError:
                    continue

        #
        # --- Strategy 3: Structured text field extraction ---
        #
        if not parsed_response:
            import re
            prediction_match = re.search(r'prediction[\s:"]*([YN]ES|NO)', clean_text, re.IGNORECASE)
            confidence_match = re.search(r'confidence[\s:"]*(\d+)', clean_text)
            risk_factors_match = re.search(r'risk_factors[\s:"]*(\[.*\])', clean_text, re.IGNORECASE)

            if prediction_match and confidence_match:
                parsed_response = {
                    "prediction": prediction_match.group(1).upper(),
                    "confidence": int(confidence_match.group(1)),
                    "original_text": response_text
                }
                if risk_factors_match:
                    try:
                        parsed_response["risk_factors"] = json.loads(risk_factors_match.group(1))
                    except json.JSONDecodeError:
                        logging.warning("Could not parse risk_factors JSON")
                logging.debug("Parsed structured text response (Strategy 3)")

        #
        # --- Strategy 4: "json_repair" fallback ---
        #
        # If we still don't have a valid parsed_response, try to repair it using json_repair.
        #
        if not parsed_response:
            repaired_text = remove_think_tags_and_triple_quotes(clean_text)
            try:
                # Attempt to parse with json_repair
                # This automatically tries to fix unquoted keys, trailing commas, etc.
                maybe_json = json_repair.loads(repaired_text)
                # If the string was super broken, maybe_json might be empty or None
                if maybe_json:
                    parsed_response = maybe_json
                    parsed_response['original_text'] = response_text
                    logging.debug("json_repair strategy succeeded in parsing as JSON (Strategy 4)")
                else:
                    logging.debug("json_repair strategy returned empty or invalid object (Strategy 4)")
            except Exception as e:
                logging.debug(f"json_repair strategy failed (Strategy 4): {e}")

        #
        # --- Strategy 5: "unstruct" fallback ---
        #
        if not parsed_response:
            unstruct_text = remove_think_tags_and_triple_quotes(clean_text)
            lines = unstruct_text.splitlines()
            lines.reverse()

            import re
            pred_pattern = re.compile(r'(prediction|rearrest_prediction)\s*:\s*(yes|no)', re.IGNORECASE)
            conf_pattern = re.compile(r'(confidence|rearrest_confidence)\s*:\s*(\d+)', re.IGNORECASE)

            prediction = None
            confidence = None

            for line in lines:
                pred_match = pred_pattern.search(line)
                if pred_match and not prediction:
                    raw_pred = pred_match.group(2).upper()
                    prediction = 'YES' if raw_pred.startswith('Y') else 'NO'

                conf_match = conf_pattern.search(line)
                if conf_match and not confidence:
                    confidence = int(conf_match.group(2))

                if prediction and confidence is not None:
                    break

            if prediction:
                parsed_response = {
                    "prediction": prediction,
                    "confidence": confidence if confidence is not None else 00,
                    "original_text": response_text
                }
                logging.debug("unstruct strategy extracted minimal data (Strategy 5)")

        # If none of the strategies worked, raise an error
        if not parsed_response:
            raise ValueError("Could not extract valid response from model output")

        #
        # --- Final Normalization ---
        #
        if not parsed_response:
            unstruct_text = remove_think_tags_and_triple_quotes(clean_text)
            lines = unstruct_text.splitlines()
            lines.reverse()

            import re
            # Only look for prediction in unstructured text
            pred_pattern = re.compile(r'(prediction|rearrest_prediction)\s*:\s*(yes|no)', re.IGNORECASE)
            
            prediction = None
            
            for line in lines:
                pred_match = pred_pattern.search(line)
                if pred_match and not prediction:
                    raw_pred = pred_match.group(2).upper()
                    prediction = 'YES' if raw_pred.startswith('Y') else 'NO'
                    break

            if prediction:
                parsed_response = {
                    "prediction": prediction,
                    "original_text": response_text
                }
                logging.debug("unstruct strategy extracted prediction (Strategy 5)")

        # If none of the strategies worked to get even a prediction, raise an error
        if not parsed_response:
            raise ValueError("Could not extract valid prediction from model output")

        #
        # --- Final Normalization ---
        #
        normalized = {
            'original_text': response_text
        }

        # Normalize 'prediction' (required)
        pred_value = str(parsed_response.get('prediction', '')).upper()
        if pred_value in ['YES', 'NO']:
            normalized['prediction'] = pred_value
        else:
            # fallback for ambiguous or partial matches
            if any(ind in pred_value for ind in ['YES', 'TRUE', '1', 'HIGH']):
                normalized['prediction'] = 'YES'
            else:
                normalized['prediction'] = 'NO'

        # Normalize 'confidence' (optional with safe fallback)
        try:
            conf_value = parsed_response.get('confidence')
            if conf_value is not None:
                # Only attempt conversion if it looks numeric
                if isinstance(conf_value, (int, float)):
                    normalized['confidence'] = int(min(max(float(conf_value), 0.0), 100.0))
                elif isinstance(conf_value, str) and conf_value.replace('.', '', 1).replace('%', '').isdigit():
                    normalized['confidence'] = int(min(max(float(conf_value.replace('%', '')), 0.0), 100.0))
                else:
                    normalized['confidence'] = 00  # Non-numeric or malformed value
            else:
                normalized['confidence'] = 00  # Missing confidence
        except (ValueError, TypeError):
            normalized['confidence'] = 00  # Any conversion error

        # If risk_factors exist, carry them over
        if 'risk_factors' in parsed_response:
            normalized['risk_factors'] = parsed_response['risk_factors']

        logging.debug(f"Normalized response: {normalized}")
        return normalized

    except Exception as e:
        logging.error(f"Error processing model: {str(model_name)}")
        logging.error(f"Error processing model response: {str(e)}")
        logging.error(f"Raw response: {response_text}")
        raise


def remove_think_tags_and_triple_quotes(text: str) -> str:
    """
    1) Remove <THINK>...</THINK> tags and all enclosed text.
    2) Remove any triple quotes.
    Returns a cleaned-up string.
    """
    import re
    # Remove <THINK> ... </THINK> (case-insensitive for the tags if needed)
    text = re.sub(r'<THINK>.*?</THINK>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove triple quotes of all forms
    triple_quotes_patterns = [r'```', r'"""', r"'''"]
    for pattern in triple_quotes_patterns:
        text = re.sub(pattern, '', text)

    return text.strip()




# --- OpenAI API Call ---
async def call_openai_api(system_message: str, prompt: str, model_name: str, config: Any) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OPENAI_API_KEY environment variable is not set.  Please set this variable in your .env file or environment.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")
            # Exit the function immediately (and the program if necessary)
            return  # Or raise an exception if you want to propagate it up
        else:
            client = OpenAI(api_key=openai_api_key)

        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            temperature=config.model_parameters["model_temperature"],
            top_p=config.model_parameters["model_top_p"],
            max_tokens=config.model_parameters["model_max_tokens"],
            messages=[
                {"role": "parser", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        duration_time = time.time() - start_time

        # Extract the raw message text and normalize the prediction via process_model_response.
        raw_message_text = response.choices[0].message.content
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        normalized_response = process_model_response(raw_message_text, model_name)
        # Normalize the prediction to uppercase for consistency.
        normalized_prediction = normalized_response.get('prediction', "").upper()
        normalized_response['prediction'] = normalized_prediction

        try:
            decision = Decision(
                prediction=normalized_prediction,
                confidence=normalized_response['confidence']
            )
        except Exception as ve:
            logging.error(f"Validation error in OpenAI API response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        extra_data = {
            'response_text': normalized_prediction,
            'raw_message_text': raw_message_text
        }
        if normalized_response.get('risk_factors'):
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,
            created_at=start_time,
            done_reason=finish_reason,
            done=True,
            total_duration=duration_time,
            load_duration=None,
            prompt_eval_count=getattr(response.usage_metadata, "prompt_token_count", None),
            prompt_eval_duration=None,
            eval_count=(getattr(response.usage_metadata, "total_token_count", 0) -
                        getattr(response.usage_metadata, "prompt_token_count", 0)),
            eval_duration=None
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


# --- Anthropic API Call ---
async def call_anthropic_api(system_message: str, prompt: str, model_name: str, config: Any) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
    try:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            logging.error("ANTHROPIC_API_KEY environment variable is not set.  Please set this variable in your .env file or environment.")
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
            # Exit the function immediately (and the program if necessary)
            return  # Or raise an exception if you want to propagate it up
        else:
            client = Anthropic(api_key=anthropic_api_key)

        start_time = time.time()
        cleaned_model_name = clean_model_name(model_name)
        response = client.messages.create(
            model=cleaned_model_name,
            temperature=config.model_parameters["model_temperature"],
            max_tokens=config.model_parameters["model_max_tokens"],
            top_p=config.model_parameters["model_top_p"],
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )
        duration_time = time.time() - start_time

        raw_message_text = response.content[0].text
        normalized_response = process_model_response(raw_message_text, model_name)
        normalized_prediction = normalized_response.get('prediction', "").upper()
        normalized_response['prediction'] = normalized_prediction

        try:
            decision = Decision(
                prediction=normalized_prediction,
                confidence=normalized_response['confidence']
            )
        except Exception as ve:
            logging.error(f"Validation error in Anthropic API response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        extra_data = {
            'response_text': normalized_prediction,
            'raw_message_text': raw_message_text
        }
        if normalized_response.get('risk_factors'):
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,
            created_at=start_time,
            done_reason=getattr(response, "stop_reason", None),
            done=True,
            total_duration=duration_time,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Anthropic API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


# --- Google API Call --

async def call_google_api(system_message: str, prompt: str, model_name: str, config: Any) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
    """
    Updated Google API implementation for Gemini models
    """
    try:
        # Initialize Google API
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logging.error("GOOGLE_API_KEY environment variable not set")
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Initialize Gemini client
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name)

        # Prepare generation config
        generation_config = {
            "temperature": config.model_parameters["model_temperature"],
            "top_p": config.model_parameters["model_top_p"],
            "max_output_tokens": config.model_parameters["model_max_tokens"]
        }

        # Combine system message and prompt
        full_prompt = f"{system_message}\n\n{prompt}"
        
        start_time = time.time()
        response = await asyncio.to_thread(
            model.generate_content,
            full_prompt,
            generation_config=generation_config
        )
        duration = time.time() - start_time

        if not response.text:
            raise ValueError("Empty response from Google API")

        # Process response
        raw_message_text = response.text
        normalized_response = process_model_response(raw_message_text, model_name)
        normalized_prediction = normalized_response.get('prediction', "").upper()

        # Create decision object
        try:
            decision = Decision(
                prediction=normalized_prediction,
                confidence=normalized_response.get('confidence', 0)
            )
        except ValidationError as ve:
            logging.error(f"Validation error in Google API response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        # Prepare metadata and extra data
        extra_data = {
            'response_text': normalized_prediction,
            'raw_message_text': raw_message_text
        }
        if normalized_response.get('risk_factors'):
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,
            created_at=start_time,
            done_reason=None,
            done=True,
            total_duration=duration,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Google API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}

async def get_decision(prompt_type: Any, api_type: str, model_name: str, config: Any, prompt: str) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
    """
    Unified decision function that handles multiple API types
    """
    try:
        # Get system message from config
        system_message = config.prompts.get("prompt_persona", "") or (
            "You are a risk assessment expert. Your responses must be in valid JSON format "
            "containing exactly three fields: 'risk_factors', 'prediction', and 'confidence'."
        )

        # API type dispatch
        if api_type == 'openai':
            return await call_openai_api(system_message, prompt, model_name, config)
        elif api_type == 'anthropic':
            return await call_anthropic_api(system_message, prompt, model_name, config)
        elif api_type == 'google':
            return await call_google_api(system_message, prompt, model_name, config)
        elif api_type == 'ollama':
            return await call_ollama_api(system_message, prompt, model_name, config)
        elif api_type == 'together':
            return await call_together_api(system_message, prompt, model_name, config)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    except Exception as e:
        logging.error(f"Error in get_decision: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}

async def get_decision_with_timeout(prompt_type: Any, api_type: str, model_name: str, config: Any, prompt: str) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any], Any]:
    """
    Wrapper function that adds timeout handling to get_decision
    """
    timeout_metrics = TimeoutMetrics()
    timeout_seconds = float(config.model_ensemble.get(model_name, {}).get("max_response_time", 30.0))

    try:
        logging.debug(f"Calling get_decision with prompt: {prompt[:50]}...")
        decision, meta_data, used_prompt, extra_data = await asyncio.wait_for(
            get_decision(prompt_type, api_type, model_name, config, prompt),
            timeout=timeout_seconds
        )
        return decision, meta_data, used_prompt, extra_data, timeout_metrics

    except asyncio.TimeoutError:
        logging.warning(f"Timeout occurred after {timeout_seconds} seconds")
        timeout_metrics.occurred = True
        timeout_metrics.retry_count += 1
        timeout_metrics.total_timeout_duration += timeout_seconds
        return None, None, prompt, {
            "error": f"Timeout after {timeout_seconds} seconds",
            "api_failure": True,
            "failure_type": "timeout"
        }, timeout_metrics

    except Exception as e:
        logging.error(f"Error in get_decision_with_timeout: {e}")
        return None, None, prompt, {
            "error": str(e),
            "api_failure": True,
            "failure_type": "api_failure"
        }, timeout_metrics



# --- save_decision Function ---
def save_decision(decision: Any, meta_data: Any, prompt_type: Any, model_name: str, row_id: int, actual_value: str, config: Any, used_prompt: str, repeat_index: int = 0, extra_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save decision and metadata to the filesystem as a JSON file.
    """
    if extra_data is None:
        extra_data = {}

    try:
        base_dir = Path(config.output["base_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        from utils import get_prompt_type_str  # Assumes this utility exists.
        prompt_type_str = get_prompt_type_str(prompt_type)
        clean_name = clean_model_name(model_name)
        model_dir = base_dir / clean_name / prompt_type_str
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if prompt_type == PromptType.COT_NSHOT:
            filename = f"{model_name}_{prompt_type_str}_id{row_id}_nshot{config.execution.nshot_ct}_{timestamp}.json"
        else:
            filename = f"{model_name}_{prompt_type_str}_id{row_id}_ver{repeat_index}_{timestamp}.json"

        output_path = model_dir / filename

        decision_data = pydantic_or_dict(decision)
        meta_data_data = pydantic_or_dict(meta_data)

        decision_data.update({
            'id': row_id,
            'actual': actual_value,
            'correct': "YES" if decision.prediction.upper() == actual_value.upper() else "NO"
        })

        meta_data_data = convert_ns_to_s(meta_data_data)

        response_data = {
            'raw_message_text': extra_data.get('raw_message_text', ''),
            'normalized_prediction': extra_data.get('response_text', '')
        }

        combined_data = {
            "decision": decision_data,
            "meta_data": meta_data_data,
            "response": response_data,
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

        combined_data["prompt"] = {
            "system": config.prompts.get("prompt_persona", ""),
            "user": used_prompt
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, default=str)

        logging.info(f"Successfully saved decision+meta_data to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {e}")
        return False
