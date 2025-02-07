# decision.py
import os
from dotenv import load_dotenv # find_dotenv
from pathlib import Path
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any, List, TypedDict
import asyncio
from datetime import datetime
from pathlib import Path
import getpass
import re # Added import for regex
from pydantic import BaseModel, Field, ValidationError
from json_repair import loads as repair_json # Added import for json_repair

# Import actual API clients - added these from reference
from ollama import chat
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
# from groq import Groq # Not adding groq or together for now, focusing on openai & anthropic
# from together import Together


from config import Config
from models import Decision, PromptType # Assuming Decision and PromptType are in models.py
from metrics import TimeoutMetrics
from utils import clean_model_name, pydantic_or_dict, convert_ns_to_s, check_existing_decision


# Load the environment variables from the .env file
# Construct path relative to the current script's location
dotenv_path = Path(__file__).parent / "src" / ".env"  # __file__ is the current file's path
load_dotenv(dotenv_path)
# load_dotenv()

API_DELAY=0.2 # Added from reference - although not directly used in all API calls in reference, good to have as a constant


# Constants for response normalization - Added from reference
MIN_POLARITY = -2
MAX_POLARITY = 2
MIN_CONFIDENCE = 0
MAX_CONFIDENCE = 100
DEFAULT_CONFIDENCE = 99


# ---------------------------
# Minimal Data Models - Updated to match reference
# ---------------------------
# In decision.py, keep Decision class as:
class Decision(BaseModel):
    prediction: int = Field(..., description="Sentiment prediction (-2 to 2)") # Keep as int for now
    confidence: int = Field(..., description="Confidence score (0 to 100)")
    reasoning: Optional[str] = Field(None, description="Model's reasoning")
    exceptions: Optional[str] = Field(None, description="Exceptions or errors reported by model")

class MetaData(BaseModel):
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

# ---------------------------
# Helper Functions - Updated process_model_response from reference
# ---------------------------
class ModelResponse(TypedDict): # Added TypedDict from reference
    prediction: int
    confidence: int
    risk_factors: Optional[List[str]]
    original_text: str
    reasoning: Optional[str]
    exceptions: Optional[str]


def remove_think_tags_and_triple_quotes(text: str) -> str:
    """Remove thinking tags and code block markers from text.""" # Docstring from reference
    # Remove triple quotes and json markers # Comment from reference
    clean_text = text.replace('`json', '').replace('`', '') # From reference
    # Remove thinking tags if present # Comment from reference
    clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL) # Regex from reference - using <think> instead of <THINK>
    return clean_text.strip() # From reference


def process_model_response(response_text: str, model_name: str) -> ModelResponse:
    """
    Process model response to extract sentiment analysis results.

    Args:
        response_text (str): Raw response from the model
        model_name (str): Name of the model used

    Returns:
        ModelResponse: Normalized response containing:
            - prediction: int (-2 to 2)
            - confidence: int (0 to 100)
            - risk_factors: list (optional)
            - original_text: str
            - reasoning: str (optional)
            - exceptions: str (optional)

    Raises:
        ValueError: If response cannot be parsed or normalized
        json.JSONDecodeError: If JSON parsing fails after all strategies
    """
    if not response_text:
        raise ValueError("Empty response text received")

    logging.debug("Processing model response", extra={
        "model_name": model_name,
        "response_length": len(response_text)
    })

    clean_text = remove_think_tags_and_triple_quotes(response_text)
    parsed_response = None

    # Strategy 1: Direct JSON parsing
    try:
        parsed_response = json.loads(clean_text)
        logging.debug("Successfully parsed response as JSON (Strategy 1)")
    except json.JSONDecodeError:
        logging.debug("Direct JSON parsing failed, trying alternative strategies")

    # Strategy 2: Extract JSON using regex
    if not parsed_response:
        matches = re.findall(r'\{.*\}', clean_text, re.DOTALL)
        for match in matches:
            try:
                parsed_response = json.loads(match)
                logging.debug("Extracted JSON using regex (Strategy 2)")
                break
            except json.JSONDecodeError:
                continue

    # Strategy 3: Fallback extraction via minimal regex
    if not parsed_response:
        pol_match = re.search(r'prediction[\s:"]*(-?[0-2])', clean_text, re.IGNORECASE)
        conf_match = re.search(r'confidence[\s:"]*(\d+)', clean_text)
        reason_match = re.search(r'reasoning[\s:"]*(.+?)(?:,|$)', clean_text, re.IGNORECASE)
        exc_match = re.search(r'exceptions[\s:"]*(.+?)(?:,|$)', clean_text, re.IGNORECASE)

        if pol_match:
            parsed_response = {
                "prediction": int(pol_match.group(1)),
                "confidence": int(conf_match.group(1)) if conf_match else DEFAULT_CONFIDENCE,
                "reasoning": reason_match.group(1).strip() if reason_match else "",
                "exceptions": exc_match.group(1).strip() if exc_match else ""
            }
            logging.debug("Extracted minimal data via regex (Strategy 3)")

    # Strategy 4: Use json_repair as a last resort
    if not parsed_response:
        try:
            parsed_response = repair_json(clean_text)
            logging.debug("Parsed response using json_repair (Strategy 4)")
        except Exception as e:
            logging.debug(f"All parsing strategies failed: {str(e)}")
            raise ValueError("Could not extract valid JSON from model response")

    # Normalize the response
    normalized: ModelResponse = {
        'original_text': response_text,
        'prediction': 0,
        'confidence': DEFAULT_CONFIDENCE,
        'risk_factors': [], # Keep risk_factors for now, even though not in reference Decision model. May be useful later.
        'reasoning': "",
        'exceptions': ""
    }

    # Normalize prediction (required, -2 to 2 range)
    prediction_value = parsed_response.get('prediction')
    if prediction_value is not None:
        try:
            if isinstance(prediction_value, str):
                prediction_value = float(prediction_value.replace('%', ''))
            normalized['prediction'] = int(min(max(float(prediction_value), MIN_POLARITY), MAX_POLARITY))
        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to parse prediction value: {prediction_value}", exc_info=e)
            normalized['prediction'] = 0

    # Normalize confidence (optional, 0 to 100 range)
    conf_value = parsed_response.get('confidence')
    if conf_value is not None:
        try:
            if isinstance(conf_value, str):
                conf_value = float(conf_value.replace('%', ''))
            normalized['confidence'] = int(min(max(float(conf_value), MIN_CONFIDENCE), MAX_CONFIDENCE))
        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to parse confidence value: {conf_value}", exc_info=e)

    # Copy over optional fields
    if 'risk_factors' in parsed_response:
        normalized['risk_factors'] = parsed_response['risk_factors']
    if 'reasoning' in parsed_response:
        normalized['reasoning'] = str(parsed_response['reasoning']).strip()
    if 'exceptions' in parsed_response:
        normalized['exceptions'] = str(parsed_response['exceptions']).strip()

    logging.debug("Normalized response", extra={"normalized": normalized})
    return normalized


# ---------------------------
# API Call Functions - Updated to add OpenAI and Anthropic, using reference style and simplified models
# ---------------------------
async def call_openai_api(system_message: str, prompt: str, model_name: str, config: Dict[str, Any]) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = OpenAI(api_key=api_key)

        start_time = time.time()
        response = None # Initialize response for different model conditions
        if model_name == 'gpt-4o-mini': # Model specific handling from reference
            response = client.chat.completions.create(
                model=model_name,
                # temperature=config.model_par  ameters.get("model_temperature"),
                temperature=config.model_parameters.get("model_temperature"),
                top_p=config.model_parameters.get("model_top_p"), #["model_parameters"]["model_top_p"],
                max_tokens=config.model_parameters.get("model_max_tokens"), #["model_parameters"]["model_max_tokens"], # Doesn't work for o3-mini API - comment from reference
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
        elif model_name == 'o3-mini-2025-01-31': # Model specific handling from reference
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type":"text"}, # From reference
                reasoning_effort="low" # From reference
            )
        else:
            print(f"ERROR: model_name: {model_name} must be in ['gpt-4o-mini','o3-mini-2025-01-03']") # Error message from reference
            pass # From reference - although maybe should raise an error?

        duration = time.time() - start_time

        raw_message_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        usage = response.usage
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else usage.prompt_tokens
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else usage.total_tokens
        eval_count = total_tokens - prompt_tokens if (prompt_tokens is not None and total_tokens is not None) else None


        normalized_response = process_model_response(raw_message_text, model_name)
        prediction_val = normalized_response.get("prediction", 0)
        confidence_val = normalized_response.get("confidence", DEFAULT_CONFIDENCE) # Using DEFAULT_CONFIDENCE
        reasoning_val = normalized_response.get("reasoning", "")
        exceptions_val = normalized_response.get("exceptions", "")

        try:
            decision = Decision(prediction=prediction_val, confidence=confidence_val, reasoning=reasoning_val, exceptions=exceptions_val)
        except ValidationError as ve:
            logging.error(f"Validation error in OpenAI response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        extra_data = {
            "response_text": raw_message_text,
            "raw_response": raw_message_text,
            "openai_metadata": usage # Save usage metadata
        }
        meta_data = MetaData(
            model=model_name,
            created_at=str(datetime.now()), # Using datetime.now() string conversion from reference
            done_reason=finish_reason,
            done=True,
            total_duration=duration,
            load_duration=None,
            prompt_eval_count=prompt_tokens,
            prompt_eval_duration=None,
            eval_count=eval_count,
            eval_duration=None
        )
        await asyncio.sleep(API_DELAY) # Added delay as in reference (although might not be strictly necessary for OpenAI/Anthropic, good practice)
        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


async def call_anthropic_api(system_message: str, prompt: str, model_name: str, config: Dict[str, Any]) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        client = Anthropic(api_key=api_key)

        start_time = time.time()
        cleaned_model_name = model_name # No clean_model_name function needed anymore as per reference
        response = client.messages.create(
            model=cleaned_model_name,
            temperature=config.model_parameters.get("model_temperature"),
            max_tokens=config.model_parameters.get("model_max_tokens"), #["model_parameters"]["model_max_tokens"],
            top_p=config.model_parameters.get("model_top_p"), #["model_parameters"]["model_top_p"],
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )
        duration = time.time() - start_time

        raw_message_text = response.content[0].text
        normalized_response = process_model_response(raw_message_text, model_name)
        prediction_val = normalized_response.get("prediction", 0)
        confidence_val = normalized_response.get("confidence", DEFAULT_CONFIDENCE) # Using DEFAULT_CONFIDENCE
        reasoning_val = normalized_response.get("reasoning", "")
        exceptions_val = normalized_response.get("exceptions", "")


        try:
            decision = Decision(prediction=prediction_val, confidence=confidence_val, reasoning=reasoning_val, exceptions=exceptions_val)
        except ValidationError as ve:
            logging.error(f"Validation error in Anthropic response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        extra_data = {"response_text": raw_message_text, "raw_response": raw_message_text}
        meta_data = MetaData(
            model=model_name,
            created_at=str(datetime.now()), # Using datetime.now() string conversion from reference
            done_reason=getattr(response, "stop_reason", None),
            done=True,
            total_duration=duration,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )
        await asyncio.sleep(API_DELAY) # Added delay as in reference
        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Anthropic API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


# --- Google API Call --- - No changes needed as google was already in the original, keeping for completeness for now. Might update later if needed.
async def call_google_api(system_message: str, prompt: str, model_name: str, config: Any) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
    try:

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logging.error("GOOGLE_API_KEY environment variable is not set.  Please set this variable in your .env file or environment.")
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
            # Exit the function immediately (and the program if necessary)
            return  # Or raise an exception if you want to propagate it up
        else:
            # client = genai.GenerativeModel(api_key=google_api_key, model_name)
            # As of 2/1/2025 these Google Gemini lower models are free, so no auth required? - Comment from original code, keeping for now, might need to revisit
            client = genai.GenerativeModel(model_name)

        start_time = time.time()
        response = client.models.generate_content(
            model=model_name,
            system_instructions=system_message,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=config.model_parameters.get("model_temperature"), #["model_parameters"]["model_temperature"],
                top_p=config.model_parameters.get("model_top_p"), #["model_top_p"],
                max_output_tokens=config.model_parameters.get("model_max_tokens"), #["model_max_tokens"]
            )
        )
        duration_time = time.time() - start_time

        raw_message_text = response.text
        normalized_response = process_model_response(raw_message_text, model_name)
        prediction_val = normalized_response.get("prediction", 0) # Updated to prediction from prediction and using prediction_val
        confidence_val = normalized_response.get("confidence", DEFAULT_CONFIDENCE) # Using DEFAULT_CONFIDENCE
        reasoning_val = normalized_response.get("reasoning", "") # Added reasoning from reference
        exceptions_val = normalized_response.get("exceptions", "") # Added exceptions from reference


        try:
            decision = Decision(prediction=prediction_val, confidence=confidence_val, reasoning=reasoning_val, exceptions=exceptions_val) # Updated Decision creation
        except Exception as ve:
            logging.error(f"Validation error in Google API response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        extra_data = {
            'response_text': raw_message_text,
            'raw_response': raw_message_text # Added raw_response for consistency
        }
        if normalized_response.get('risk_factors'): # Keeping risk_factors
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,
            created_at=str(datetime.now()), # Using datetime.now() string conversion from reference
            done_reason=None, # From reference
            done=True,
            total_duration=duration_time,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )
        await asyncio.sleep(API_DELAY) # Added delay as in reference, though might not be needed for google.
        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Google API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


# --- Unified get_decision Function --- - Updated to include openai and anthropic calls, and use updated config as Dict
async def get_decision(prompt_type: Any, api_type: str, model_name: str, config: Dict[str, Any], prompt: str) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]: # Updated type hints and config type
    """
    Dispatch the API call based on api_type.
    Returns a 4-tuple: (Decision, MetaData, used_prompt, extra_data).
    In case of failure, extra_data includes an "api_failure" flag and a "failure_type".
    """
    try:
        # system_message = config["prompts"].get("SYSTEM_PROMPT", "") 
        system_message = config.prompts.get("SYSTEM_PROMPT", "") or ( # Using "SYSTEM_PROMPT" key from reference
            "You are a risk assessment expert. Your responses must be in valid JSON format "
            "containing exactly three fields: 'risk_factors', 'prediction', and 'confidence'." # Keeping original fallback system message for now, might update if needed
        )

        api_type = api_type.lower() # Added lowercasing api_type from reference

        if api_type == 'openai':
            return await call_openai_api(system_message, prompt, model_name, config)
        elif api_type == 'anthropic':
            return await call_anthropic_api(system_message, prompt, model_name, config)
        elif api_type == 'google':
            return call_google_api(system_message, prompt, model_name, config)
        elif api_type == 'ollama':
            response = await asyncio.to_thread(
                chat,
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt}
                ],
                model=model_name,
                options={
                    'temperature': config.model_parameters.get("model_temperature"), #["model_parameters"]["model_temperature"], # Using dict-style config access
                    'top_p': config.model_parameters.get("model_top_p"), #["model_parameters"]["model_top_p"], # Using dict-style config access
                    'max_tokens': config.model_parameters.get("model_max_tokens"), #["model_parameters"]["model_max_tokens"], # Using dict-style config access
                }
            )

            if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
                raise ValueError("Invalid API response structure")

            raw_message_text = response.message.content
            print(f'raw_message_text: {raw_message_text}') # DEBUG
            normalized_response = process_model_response(raw_message_text, model_name)
            print(f"normalized_response: {normalized_response}") # DEBUG
            prediction_val = normalized_response.get("prediction", 0) # Updated to prediction from prediction and using prediction_val
            confidence_val = normalized_response.get("confidence", DEFAULT_CONFIDENCE) # Using DEFAULT_CONFIDENCE
            reasoning_val = normalized_response.get("reasoning", "") # Added reasoning from reference
            exceptions_val = normalized_response.get("exceptions", "") # Added exceptions from reference


            try:
                decision = Decision(prediction=prediction_val, confidence=confidence_val, reasoning=reasoning_val, exceptions=exceptions_val) # Updated Decision creation
            except Exception as ve:
                logging.error(f"Validation error in Ollama API response: {ve}")
                return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

            extra_data = {
                'response_text': raw_message_text,
                'raw_response': raw_message_text # Added raw_response for consistency
            }
            if normalized_response.get('risk_factors'): # Keeping risk_factors
                extra_data['risk_factors'] = normalized_response['risk_factors']

            meta_data = MetaData(
                model=getattr(response, 'model', None),
                created_at=str(datetime.now()), # Using datetime.now() string conversion from reference
                done_reason=getattr(response, 'done_reason', None),
                done=getattr(response, 'done', None),
                total_duration=getattr(response, 'total_duration', None),
                load_duration=getattr(response, 'load_duration', None),
                prompt_eval_count=getattr(response, 'prompt_eval_count', None),
                prompt_eval_duration=getattr(response, 'prompt_eval_duration', None),
                eval_count=getattr(response, 'eval_count', None),
                eval_duration=getattr(response, 'eval_duration', None)
            )
            await asyncio.sleep(API_DELAY) # Added delay as in reference, though likely not needed for local ollama.

            return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Error during API call in get_decision: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


async def get_decision_with_timeout(prompt_type: Any, api_type: str, model_name: str, config: Dict[str, Any], prompt: str) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any], Any]: # Updated config type to Dict[str, Any]
    """
    Wrap get_decision with timeout handling.
    Returns (Decision, MetaData, used_prompt, extra_data, TimeoutMetrics).
    """
    timeout_metrics = TimeoutMetrics(occurred=False, retry_count=0, total_timeout_duration=0.0)
    # timeout_seconds = float(config.get("api_timeout", 30)) # Using config.get("api_timeout", 30) from reference
    timeout_seconds = float(config.timeout.get("api_timeout", 30)) # Corrected approach: access 'timeout' dictionary, then use .get()

    try:
        logging.debug(f"Calling get_decision with prompt: {prompt[:50]}...")
        decision, meta_data, used_prompt, extra_data = await asyncio.wait_for(
            get_decision(prompt_type, api_type, model_name, config, prompt),
            timeout=timeout_seconds
        )
        logging.debug("Received response from get_decision.")
        return decision, meta_data, used_prompt, extra_data, timeout_metrics

    except asyncio.TimeoutError:
        logging.warning("API call timed out while waiting for model response...")
        timeout_metrics.occurred = True
        timeout_metrics.retry_count += 1
        return None, None, prompt, {"error": "Timeout", "api_failure": True, "failure_type": "timeout"}, timeout_metrics

    except Exception as e:
        logging.error(f"Error during API call in get_decision_with_timeout: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}, timeout_metrics





# --- save_decision Function --- - save_decision function remains mostly the same as it was already quite comprehensive in original code
# In decision.py, update save_decision function:
def save_decision(decision: Any, meta_data: Any, prompt_type: Any, model_name: str, row_id: int, actual_value: str, config: Any, used_prompt: str, repeat_index: int = 0, extra_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save decision and metadata to the filesystem as a JSON file.
    """
    if extra_data is None:
        extra_data = {}

    try:
        base_dir = Path(config.output["base_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        from utils import get_prompt_type_str
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

        decision_data = pydantic_or_dict(decision) # Get dict from Pydantic Decision model
        meta_data_data = pydantic_or_dict(meta_data) # Get dict from Pydantic MetaData model

        # --- Restructure decision_data to match desired output ---
        decision_output = { # Create a new dict for 'decision' output section
            "prediction": "YES" if decision.prediction >= 0 else "NO", # Convert numerical prediction to "YES"/"NO" string
            "confidence": decision.confidence,
            "id": row_id,
            "actual": actual_value,
            "correct": "YES" if decision.prediction >= 0 and actual_value.upper() == "YES" or decision.prediction < 0 and actual_value.upper() == "NO" else "NO"
        }
        # No need to update decision_data anymore, use decision_output in combined_data

        meta_data_data = convert_ns_to_s(meta_data_data)

        response_data = {
            'raw_message_text': extra_data.get('raw_message_text', ''), # Keep raw message text
            'normalized_prediction': extra_data.get('response_text', '') # Keep normalized_prediction as response_text for now - will revisit
        }

        combined_data = {
            "decision": decision_output, # Use the restructured decision_output here
            "meta_data": meta_data_data,
            "response": response_data,
            "evaluation": {
                "timestamp": timestamp,
                "model": model_name,
                "prompt_type": str(prompt_type),
                "row_id": row_id,
                "prediction_matches_actual": decision_output['correct'], # Use from decision_output
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