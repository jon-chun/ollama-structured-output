# decision.py
import os
import sys
from dotenv import load_dotenv # find_dotenv
from pathlib import Path
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any, List, TypedDict, Union
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
from groq import Groq # Not adding groq or together for now, focusing on openai & anthropic
from together import Together

from config import load_config, Config
# import models
# print("Models module path:", models.__file__)
from models import Decision, PromptType
print("Decision model source:", Decision.__module__)
print("Decision model mro:", Decision.__mro__)

print("Decision class type:", type(Decision))
print("Decision class schema:", Decision.model_json_schema())  # Better way to inspect Pydantic model

# sys.path.insert(0, str(Path(__file__).parent))  # Add current directory to path
# from models import Decision, PromptType  # This will now find the local version first
# import models
# print("Models module path:", models.__file__)

from src.metrics import TimeoutMetrics
from src.utils import clean_model_name, pydantic_or_dict, convert_ns_to_s, check_existing_decision

# print("Decision class location:", Decision.__file__)
print("Decision class definition:", Decision.__dict__)

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
        'prediction': "UNKNOWN",  # Default to "UNKNOWN" instead of 0
        'confidence': DEFAULT_CONFIDENCE,
        'risk_factors': [],
        'reasoning': "",
        'exceptions': ""
    }

    # New prediction normalization logic
    prediction_value = parsed_response.get('prediction')
    if prediction_value is not None:
        pred_str = str(prediction_value).upper()
        if pred_str in ["YES", "NO", "UNKNOWN"]:
            normalized['prediction'] = pred_str
        else:
            logging.warning(f"Unexpected prediction value type: {type(prediction_value)}")

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
def create_validated_decision(
    normalized_response: Dict[str, Any], 
    raw_message_text: str,
    model_specific_data: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Decision], Dict[str, Any]]:
    """Enhanced helper function to handle model-specific data"""
    # Validation logic same as before
    prediction_val = normalized_response.get("prediction", "NO")
    prediction_val = str(prediction_val).upper()
    if prediction_val not in ["YES", "NO"]:
        prediction_val = "NO"

    # Validate confidence
    try:
        confidence_val = int(normalized_response.get("confidence", DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        confidence_val = DEFAULT_CONFIDENCE
    confidence_val = min(max(0, confidence_val), 100)

    logging.debug(f"Creating Decision with values: prediction={prediction_val!r}, confidence={confidence_val}")
    try:
        decision = Decision(
            prediction=prediction_val,
            confidence=confidence_val
        )
        logging.debug(f"Successfully created Decision: {decision.model_dump()}")
        
        # Create extra_data
        extra_data = {
            'response_text': prediction_val,
            'raw_message_text': raw_message_text
        }
        # Add any model specific data
        if normalized_response.get('risk_factors'):
            extra_data['risk_factors'] = normalized_response['risk_factors']
            
        return decision, extra_data
        
    except Exception as ve:
        logging.error(f"Validation error: {ve}")
        logging.error(f"Input values - prediction: {prediction_val} ({type(prediction_val)}), confidence: {confidence_val} ({type(confidence_val)})")
        return None, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

def create_metadata(
    response: Any, 
    model_name: str,
    duration: Optional[float] = None,
    **kwargs
) -> MetaData:
    """Helper function to create MetaData with flexible fields"""
    return MetaData(
        model=model_name,
        created_at=str(datetime.now()),
        done_reason=kwargs.get('done_reason') or getattr(response, 'done_reason', None),
        done=kwargs.get('done', True),
        total_duration=duration or getattr(response, 'total_duration', None),
        load_duration=kwargs.get('load_duration') or getattr(response, 'load_duration', None),
        prompt_eval_count=kwargs.get('prompt_eval_count') or getattr(response, 'prompt_eval_count', None),
        prompt_eval_duration=kwargs.get('prompt_eval_duration') or getattr(response, 'prompt_eval_duration', None),
        eval_count=kwargs.get('eval_count') or getattr(response, 'eval_count', None),
        eval_duration=kwargs.get('eval_duration') or getattr(response, 'eval_duration', None)
    )


async def call_ollama_api(system_message: str, prompt: str, model_name: str, config: Dict[str, Any]) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    """Call Ollama API with consistent interface matching other API calls."""
    try:
        start_time = time.time()
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
        duration = time.time() - start_time

        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            raise ValueError("Invalid API response structure")

        raw_message_text = response.message.content
        logging.debug(f'raw_message_text: {raw_message_text}')
        
        normalized_response = process_model_response(raw_message_text, model_name)
        logging.debug(f"normalized_response: {normalized_response}")

        decision, extra_data = create_validated_decision(normalized_response, raw_message_text)
        if not decision:
            return None, None, prompt, extra_data

        meta_data = create_metadata(
            response,
            model_name,
            duration=duration
        )

        await asyncio.sleep(API_DELAY)
        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Ollama API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}
    

async def call_openai_api(system_message: str, prompt: str, model_name: str, config: Dict[str, Any]) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        # API setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = OpenAI(api_key=api_key)

        # Model-specific options
        model_options = {
            'gpt-4o-mini': {
                'temperature': config.model_parameters.get("model_temperature"),
                'top_p': config.model_parameters.get("model_top_p"),
                'max_tokens': config.model_parameters.get("model_max_tokens")
            },
            'o3-mini-2025-01-31': {
                'response_format': {"type": "text"},
                'reasoning_effort': "low"
            }
        }

        if model_name not in model_options:
            print(f"ERROR: model_name: {model_name} must be in ['gpt-4o-mini','o3-mini-2025-01-03']") # Error message from reference
            raise ValueError(f"Unsupported model: {model_name}")

        # API call with model-specific options
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            **model_options[model_name]
        )
        duration = time.time() - start_time

        raw_message_text = response.choices[0].message.content

        logging.debug(f'raw_message_text: {raw_message_text}')
        
        normalized_response = process_model_response(raw_message_text, model_name)
        logging.debug(f"normalized_response: {normalized_response}")

        # Use helper functions
        decision, extra_data = create_validated_decision(normalized_response, raw_message_text)
        if not decision:
            return None, None, prompt, extra_data
            
        # Add OpenAI-specific data
        extra_data["openai_metadata"] = response.usage
        
        meta_data = create_metadata(
            response,
            model_name,
            duration,
            done_reason=response.choices[0].finish_reason,
            prompt_eval_count=response.usage.prompt_tokens,
            eval_count=response.usage.total_tokens - response.usage.prompt_tokens
        )
        
        return decision, meta_data, prompt, extra_data
        
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


async def call_anthropic_api(system_message: str, prompt: str, model_name: str, config: Config) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        # Ensure the Anthropic API key is set
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
            
        # Initialize the Anthropic client
        client = Anthropic(api_key=api_key)

        # Start timer for performance measurement
        start_time = time.time()

        # Make the API call with model-specific options using attribute access for config
        response = client.messages.create(
            model=model_name,
            temperature=config.model_parameters.get("model_temperature"),
            max_tokens=config.model_parameters.get("model_max_tokens"),
            top_p=config.model_parameters.get("model_top_p"),
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )
        duration = time.time() - start_time

        # Extract raw message text from the response
        raw_message_text = response.content[0].text
        logging.debug(f"raw_message_text: {raw_message_text}")

        # Process and normalize the model response
        normalized_response = process_model_response(raw_message_text, model_name)
        logging.debug(f"normalized_response: {normalized_response}")

        # Use the shared helper to create a validated Decision.
        # This ensures that the 'prediction' is a string ("YES" or "NO") and 'confidence' is within 0-100.
        decision, extra_data = create_validated_decision(normalized_response, raw_message_text)
        if not decision:
            return None, None, prompt, extra_data

        # Create metadata similar to the OpenAI implementation.
        # Note: Anthropic responses may have a 'stop_reason' attribute.
        meta_data = create_metadata(
            response,
            model_name,
            duration,
            done_reason=getattr(response, "stop_reason", None)
        )

        # Optionally add any Anthropic-specific metadata to extra_data.
        extra_data["anthropic_metadata"] = getattr(response, "usage", {})

        # Optional delay (if your design requires pacing between API calls)
        await asyncio.sleep(API_DELAY)
        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Anthropic API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


# --- Google API Call --- - No changes needed as google was already in the original, keeping for completeness for now. Might update later if needed.
async def call_google_api(system_message: str, prompt: str, model_name: str, config: Config) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        # Ensure Google API key is present
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logging.error("GOOGLE_API_KEY environment variable is not set. Please set this variable in your .env file or environment.")
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        # Initialize Google API client
        client = genai.Client(api_key=google_api_key)

        # Merge system instructions with the user prompt (since Gemini API lacks structured roles)
        contents = f"SYSTEM: {system_message}\nUSER: {prompt}"

        # Configure generation parameters using attribute access
        generation_config = types.GenerateContentConfig(
            temperature=config.model_parameters.get("model_temperature"),
            top_p=config.model_parameters.get("model_top_p"),
            max_output_tokens=config.model_parameters.get("model_max_tokens")
        )

        # Perform the API call and measure duration
        start_time = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generation_config
        )
        duration_time = time.time() - start_time

        # Extract generated text
        raw_message_text = response.candidates[0].content.parts[0].text
        logging.debug(f"raw_message_text: {raw_message_text}")

        # Process the model response using your existing parser
        normalized_response = process_model_response(raw_message_text, model_name)
        logging.debug(f"normalized_response: {normalized_response}")

        # Extract values from the normalized response
        prediction_val = normalized_response.get("prediction", 0)
        confidence_val = normalized_response.get("confidence", DEFAULT_CONFIDENCE)
        reasoning_val = normalized_response.get("reasoning", "")
        exceptions_val = normalized_response.get("exceptions", "")

        # Attempt to create a Decision object (update as needed if your Decision model supports extra fields)
        try:
            decision = Decision(
                prediction=prediction_val,
                confidence=confidence_val,
                reasoning=reasoning_val,
                exceptions=exceptions_val
            )
        except Exception as ve:
            logging.error(f"Validation error in Google API response: {ve}")
            return None, None, prompt, {"error": str(ve), "api_failure": True, "failure_type": "parse_failure"}

        # Construct extra data with raw response details and any risk factors
        extra_data = {
            'response_text': raw_message_text,
            'raw_response': raw_message_text
        }
        if normalized_response.get('risk_factors'):
            extra_data['risk_factors'] = normalized_response['risk_factors']

        # Create MetaData for this API call
        meta_data = MetaData(
            model=model_name,
            created_at=str(datetime.now()),
            done_reason=None,
            done=True,
            total_duration=duration_time,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )

        # Optionally delay further API calls if needed
        await asyncio.sleep(API_DELAY)
        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Google API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}

async def call_groq_api(system_message: str, prompt: str, model_name: str, config: Config) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        client = Groq(api_key=api_key)

        # Optional delay to pace API calls
        await asyncio.sleep(API_DELAY)
        start_time = time.time()

        # Select the appropriate API call based on the model name
        if model_name in ['deepseek-r1-distill-llama-70b', 'llama3-70b-versatile']:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.model_parameters.get("model_temperature"),
                top_p=config.model_parameters.get("model_top_p"),
                max_completion_tokens=config.model_parameters.get("model_max_tokens"),
                stream=False,
                stop=None
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}. Must be one of ['deepseek-r1-distill-llama-70b', 'llama3-70b-versatile']")
        
        duration = time.time() - start_time

        # Extract the raw text and finish reason from the response
        raw_message_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # Extract usage metadata safely (supporting both dicts and attribute access)
        usage = response.usage
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else usage.prompt_tokens
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else usage.total_tokens
        eval_count = total_tokens - prompt_tokens if (prompt_tokens is not None and total_tokens is not None) else None

        # Process and normalize the model response
        normalized_response = process_model_response(raw_message_text, model_name)
        logging.debug(f"normalized_response (Groq): {normalized_response}")

        # Create a validated decision using the helper; this ensures the 'prediction' is "YES"/"NO"
        decision, extra_data = create_validated_decision(normalized_response, raw_message_text)
        if not decision:
            return None, None, prompt, extra_data

        # Add Groq-specific metadata to the extra_data
        extra_data["groq_metadata"] = usage

        # Create metadata using the shared helper function
        meta_data = create_metadata(
            response,
            model_name,
            duration,
            done_reason=finish_reason,
            prompt_eval_count=prompt_tokens,
            eval_count=eval_count
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Groq API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}


async def call_together_api(system_message: str, prompt: str, model_name: str, config: Config) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set.")
        client = Together(api_key=api_key)

        # Optional delay to pace API calls
        await asyncio.sleep(API_DELAY)
        start_time = time.time()

        # Branch based on the model name to apply model-specific options
        if model_name == 'DeepSeek-V3':
            response = client.chat.completions.create(
                model=f"deepseek-ai/{model_name}",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.model_parameters.get("model_max_tokens"),
                temperature=config.model_parameters.get("model_temperature"),
                top_p=config.model_parameters.get("model_top_p"),
                repetition_penality=config.model_parameters.get("repetition_penality"),
                stop=["<｜end▁of▁sentence｜>"],
                stream=False,
            )
        elif model_name == 'Qwen2.5-72B-Instruct-Turbo':
            response = client.chat.completions.create(
                model=f"Qwen/{model_name}",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.model_parameters.get("model_max_tokens"),
                temperature=config.model_parameters.get("model_temperature"),
                top_p=config.model_parameters.get("model_top_p"),
                repetition_penality=config.model_parameters.get("repetition_penality"),
                stop=["<｜end▁of▁sentence｜>"],
                stream=False,
            )
        elif model_name == 'DeepSeek-R1-Distill-Llama-70B':
            response = client.chat.completions.create(
                model=f"deepseek-ai/{model_name}",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.model_parameters.get("model_max_tokens"),
                temperature=config.model_parameters.get("model_temperature"),
                top_p=config.model_parameters.get("model_top_p"),
                repetition_penality=config.model_parameters.get("repetition_penality"),
                stop=["<｜end▁of▁sentence｜>"],
                stream=False,
            )
        elif model_name == 'Qwen2-VL-72B-Instruct':
            response = client.chat.completions.create(
                model=f"Qwen/{model_name}",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.model_parameters.get("model_max_tokens"),
                temperature=config.model_parameters.get("model_temperature"),
                top_p=config.model_parameters.get("model_top_p"),
                # repetition_penality omitted if not applicable
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=False,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}. Must be one of ['DeepSeek-V3', 'Qwen2.5-72B-Instruct-Turbo', 'DeepSeek-R1-Distill-Llama-70B', 'Qwen2-VL-72B-Instruct']")

        duration = time.time() - start_time

        # Extract the raw text and finish reason from the response
        raw_message_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # Extract usage metadata with safe access
        usage = response.usage
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else usage.prompt_tokens
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else usage.total_tokens
        eval_count = total_tokens - prompt_tokens if (prompt_tokens is not None and total_tokens is not None) else None

        # Process and normalize the model response
        normalized_response = process_model_response(raw_message_text, model_name)
        logging.debug(f"normalized_response (Together): {normalized_response}")

        # Create a validated Decision (with "YES"/"NO" prediction) using the shared helper function
        decision, extra_data = create_validated_decision(normalized_response, raw_message_text)
        if not decision:
            return None, None, prompt, extra_data

        # Add Together-specific metadata to extra_data
        extra_data["together_metadata"] = usage

        # Create metadata using the shared helper
        meta_data = create_metadata(
            response,
            model_name,
            duration,
            done_reason=finish_reason,
            prompt_eval_count=prompt_tokens,
            eval_count=eval_count
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Together API call failed: {e}")
        return None, None, prompt, {"error": str(e), "api_failure": True, "failure_type": "api_failure"}



# --- Unified get_decision Function --- - Updated to include openai and anthropic calls, and use updated config as Dict
async def get_decision(prompt_type: Any, api_type: str, model_name: str, config: Dict[str, Any], prompt: str) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]:
    """
    Dispatch the API call based on api_type.
    Returns a 4-tuple: (Decision, MetaData, used_prompt, extra_data).
    In case of failure, extra_data includes an "api_failure" flag and a "failure_type".
    """
    try:
        system_message = config.prompts.get("SYSTEM_PROMPT", "") or (
            "You are a risk assessment expert. Your responses must be in valid JSON format "
            "containing exactly three fields: 'risk_factors', 'prediction', and 'confidence'."
        )

        api_type = api_type.lower()

        # Map API types to their handler functions
        api_handlers = {
            'openai': call_openai_api,
            'anthropic': call_anthropic_api,
            'google': call_google_api,
            'groq': call_groq_api,
            'together': call_together_api,
            'ollama': call_ollama_api
        }

        handler = api_handlers.get(api_type)
        if not handler:
            raise ValueError(f"Unsupported API type: {api_type}")

        return await handler(system_message, prompt, model_name, config)

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
        def normalize_prediction(pred: Union[str, int]) -> str:
            if isinstance(pred, str):
                pred_upper = pred.upper()
                if pred_upper == "YES" or pred_upper == "NO":
                    return pred_upper
            return "UNKNOWN"

        def is_matching_prediction(pred: Union[str, int], actual: Union[str, int]) -> str:
            norm_pred = normalize_prediction(pred)
            norm_actual = normalize_prediction(actual)
            if norm_pred == "UNKNOWN" or norm_actual == "UNKNOWN":
                return "NO"  # Conservative approach - unknown predictions count as incorrect
            return "YES" if norm_pred == norm_actual else "NO"

        decision_output = {
            'prediction': normalize_prediction(decision.prediction),
            "confidence": decision.confidence,
            "id": row_id,
            "actual": actual_value,
            'correct': is_matching_prediction(decision.prediction, actual_value)
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