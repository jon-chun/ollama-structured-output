# decision.py

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

# Make sure you have installed json-repair:
#   pip install json-repair
import json_repair

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
                    "confidence": confidence if confidence is not None else 90,
                    "original_text": response_text
                }
                logging.debug("unstruct strategy extracted minimal data (Strategy 5)")

        # If none of the strategies worked, raise an error
        if not parsed_response:
            raise ValueError("Could not extract valid response from model output")

        #
        # --- Final Normalization ---
        #
        normalized = {
            'original_text': response_text
        }

        # Normalize 'prediction'
        pred_value = str(parsed_response.get('prediction', '')).upper()
        if pred_value in ['YES', 'NO']:
            normalized['prediction'] = pred_value
        else:
            # fallback for ambiguous or partial matches
            if any(ind in pred_value for ind in ['YES', 'TRUE', '1', 'HIGH']):
                normalized['prediction'] = 'YES'
            else:
                normalized['prediction'] = 'NO'

        # Normalize 'confidence'
        conf_value = parsed_response.get('confidence', None)
        if conf_value is not None:
            if isinstance(conf_value, str):
                conf_value = float(conf_value.replace('%', ''))
            normalized['confidence'] = int(min(max(float(conf_value), 0.0), 100.0))
        else:
            normalized['confidence'] = 90  # Default

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



async def call_openai_api(system_message, prompt, model_name, config):
    try:

        client = OpenAI(getpass.getpass('Please enter your OpenAI API Key: '))

        start_time = time.time()  # Record the start time
        response = client.chat.completions.create(
            model=model_name,
            temperature=config.model_parameters["model_temperature"],
            top_p=config.model_parameters["model_top_p"],
            max_tokens=config.model_parameters["model_max_tokens"],
            messages=[
                {
                    "role": "parser", 
                    "content": system_message
                    },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        duration_time = time.time() - start_time  # Calculate total duration

        # Extract message content from OpenAI response
        # raw_message_text = response["choices"][0]["message"]["content"]
        raw_message_text = response.choices[0].message.content

        # Process response (assuming process_model_response is defined)
        normalized_response = process_model_response(raw_message_text, model_name)

        try:
            decision = Decision(
                prediction=normalized_response['prediction'],
                confidence=normalized_response['confidence']
            )
        except ValidationError as ve:
            logging.error(f"Validation error creating Decision: {str(ve)}")
            logging.error(f"Normalized response: {normalized_response}")
            return None, None, prompt, {}

        extra_data = {
            'response_text': normalized_response['prediction'],
            'raw_message_text': raw_message_text  # Add raw message to extra_data
        }
        if 'risk_factors' in normalized_response and normalized_response['risk_factors']:
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,
            created_at=start_time,
            done_reason=response["choices"][0].get("finish_reason"),
            done=True,  # OpenAI API returns 'choices', assume done when response is received
            total_duration=duration_time,  # OpenAI does not provide total duration directly
            load_duration=None,
            prompt_eval_count=response.usage_metadata.prompt_token_count,
            prompt_eval_duration=None,
            eval_count=response.usage_metadata.total_token_count-response.usage_metadata.prompt_token_count,
            eval_duration=None
        )

        return decision, meta_data, prompt, extra_data
    
    except Exception as e:
        logging.error(f"OpeanAI API call failed: {str(e)}")
        return None, None, prompt, {}

async def call_anthropic_api(system_message, prompt, model_name, config):
    try:
        client = anthropic.Anthropic(api_key=getpass.getpass("Enter Anthropic API Key: "))

        start_time = time.time()  # Record the start time
        response = client.messages.create(
            model=clean_model_name,
            temperature=config.model_parameters["model_temperature"],
            max_tokens=config.model_parameters["model_max_tokens"],
            top_p=config.model_parameters["model_top_p"],
            system=system_message,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        duration_time = time.time() - start_time  # Calculate total duration

        # Extract response text
        raw_message_text = response.content[0].text  # Anthropic returns content as a list

        # Process response (assuming process_model_response is defined)
        normalized_response = process_model_response(raw_message_text, model_name)

        try:
            decision = Decision(
                prediction=normalized_response['prediction'],
                confidence=normalized_response['confidence']
            )
        except ValidationError as ve:
            logging.error(f"Validation error creating Decision: {str(ve)}")
            logging.error(f"Normalized response: {normalized_response}")
            return None, None, prompt, {}

        extra_data = {
            'response_text': normalized_response['prediction'],
            'raw_message_text': raw_message_text  # Add raw message to extra_data
        }
        if 'risk_factors' in normalized_response and normalized_response['risk_factors']:
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,
            created_at=start_time,  # Anthropic API does not return created_at timestamp
            done_reason=response.stop_reason,  # Anthropic uses 'stop_reason'
            done=True,  # Assuming done when response is received
            total_duration=duration_time,  # Anthropic does not provide total duration directly
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Anthropic API call failed: {str(e)}")
        return None, None, prompt, {}


def call_google_api(system_message, prompt, model_name, config):
    try:
        # Initialize Google GenAI Client
        client = genai.GenerativeModel(model_name)

        # Generate response
        start_time = time.time()

        response = client.models.generate_content(
            model=model_name,
            system_instructions=system_message,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature = config.model_parameters["model_temperature"],
                top_p = config.model_parameters["model_top_p"],
                max_output_tokens = config.model_parameters["model_max_tokens"]
            )
        )
        duration_time = time.time() - start_time  # Calculate total duration

        # Extract response text
        raw_message_text = response.text  # Google’s response format


     

        response = client.models.generate_content(
            model=model_name,
            messages=[
                {"role": "system", "parts": [{"text": system_message}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            options={
                "temperature": config.model_parameters["model_temperature"],
                "top_p": config.model_parameters["model_top_p"],
                "max_output_tokens": config.model_parameters["model_max_tokens"]
            }
        )


        # Process response (assuming process_model_response is defined)
        normalized_response = process_model_response(raw_message_text, model_name)

        try:
            decision = Decision(
                prediction=normalized_response['prediction'],
                confidence=normalized_response['confidence']
            )
        except ValidationError as ve:
            logging.error(f"Validation error creating Decision: {str(ve)}")
            logging.error(f"Normalized response: {normalized_response}")
            return None, None, prompt, {}

        extra_data = {
            'response_text': normalized_response['prediction'],
            'raw_message_text': raw_message_text  # Add raw message to extra_data
        }
        if 'risk_factors' in normalized_response and normalized_response['risk_factors']:
            extra_data['risk_factors'] = normalized_response['risk_factors']

        meta_data = MetaData(
            model=model_name,  # Google API does not explicitly return model in response
            created_at=start_time,  # Google API does not return a creation timestamp
            done_reason=None,  # Google does not provide a direct equivalent of `stop_reason`
            done=True,  # Assuming done when response is received
            total_duration=duration_time,  # Google API does not provide total duration directly
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )

        return decision, meta_data, prompt, extra_data

    except Exception as e:
        logging.error(f"Google API call failed: {str(e)}")
        return None, None, prompt, {}


async def get_decision(
    prompt_type: PromptType,
    api_type: str,
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

        if api_type == 'openai':
            decision, meta_data, prompt, extra_data = call_openai_api(system_message, prompt, model_name, config)
            return decision, meta_data, prompt, extra_data 
        
        elif api_type == 'anthropic':
            decision, meta_data, prompt, extra_data = call_anthropic_api(system_message, prompt, model_name, config)
            return decision, meta_data, prompt, extra_data
        
        elif api_type == 'google':
            decision, meta_data, prompt, extra_data = call_google_api(system_message, prompt, model_name, config)
            return decision, meta_data, prompt, extra_data
            
        elif api_type == 'ollama':
        
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

            # Store the raw message text
            raw_message_text = response.message.content

            normalized_response = process_model_response(raw_message_text, model_name)

            try:
                decision = Decision(
                    prediction=normalized_response['prediction'],
                    confidence=normalized_response['confidence']
                )
            except ValidationError as ve:
                logging.error(f"Validation error creating Decision: {str(ve)}")
                logging.error(f"Normalized response: {normalized_response}")
                return None, None, prompt, {}

            extra_data = {
                'response_text': normalized_response['prediction'],
                'raw_message_text': raw_message_text  # Add raw message to extra_data
            }
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
    api_type: str,
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
            get_decision(prompt_type, api_type, model_name, config, prompt),
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
        # Create base output directory (evaluation_results)
        base_dir = Path(config.output["base_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        # Get prompt type string using our utility function
        from utils import get_prompt_type_str
        prompt_type_str = get_prompt_type_str(prompt_type)
        
        # Generate clean model name
        clean_name = clean_model_name(model_name)
        
        # Create the complete output directory structure
        model_dir = base_dir / clean_name / prompt_type_str
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Generate filename (not full path)
        if prompt_type == PromptType.COT_NSHOT:
            filename = (
                f"{model_name}_{prompt_type_str}_id{row_id}_"
                f"nshot{config.execution.nshot_ct}_{timestamp}.json"
            )
        else:
            filename = (
                f"{model_name}_{prompt_type_str}_id{row_id}_"
                f"ver{repeat_index}_{timestamp}.json"
            )
        
        # Combine directory and filename
        output_path = model_dir / filename

        # Prepare the data to save
        decision_data = pydantic_or_dict(decision)
        meta_data_data = pydantic_or_dict(meta_data)

        decision_data.update({
            'id': row_id,
            'actual': actual_value,
            'correct': "YES" if decision.prediction.upper() == actual_value.upper() else "NO"
        })

        meta_data_data = convert_ns_to_s(meta_data_data)

        # Create response object with raw message text
        response_data = {
            'raw_message_text': extra_data.get('raw_message_text', ''),
            'normalized_prediction': extra_data.get('response_text', '')
        }

        combined_data = {
            "decision": decision_data,
            "meta_data": meta_data_data,
            "response": response_data,  # Add response data
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
        
        # Save both the prompts and raw response
        combined_data["prompt"] = {
            "system": config.prompts.get("prompt_persona", ""),
            "user": used_prompt
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, default=str)

        logging.info(f"Successfully saved decision+meta_data to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {str(e)}")
        return False