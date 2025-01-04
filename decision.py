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
from utils import pydantic_or_dict, convert_ns_to_s

class MetaData(BaseModel):
    """Model to represent metadata from an API response"""
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

'''
class MetaData(BaseModel):
    """Model to represent metadata from an API response"""
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

    class Config:
        validate_assignment = True
''';

def extract_meta_data(response: Any) -> MetaData:
    """
    Extract metadata from API response.
    
    This function safely extracts metadata fields from the API response,
    converting time measurements from nanoseconds to seconds where appropriate.
    
    Args:
        response: Raw API response object
        
    Returns:
        MetaData object containing extracted information
    """
    meta_data = {}
    
    # Extract basic fields
    meta_data["model"] = getattr(response, "model", None)
    meta_data["created_at"] = getattr(response, "created_at", None)
    meta_data["done_reason"] = getattr(response, "done_reason", None)
    meta_data["done"] = getattr(response, "done", None)
    
    # Extract and convert timing fields
    timing_fields = [
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration"
    ]
    
    for field in timing_fields:
        value = getattr(response, field, None)
        if value is not None:
            # Convert nanoseconds to seconds if needed
            meta_data[field] = float(value) / 1e9 if value > 1e7 else float(value)
    
    # Extract count fields
    count_fields = ["prompt_eval_count", "eval_count"]
    for field in count_fields:
        meta_data[field] = getattr(response, field, None)
    
    return MetaData(**meta_data)


def process_model_response(response_text: str) -> Dict:
    """
    Process and normalize the model's response text into a standard format.
    
    This function handles various response formats and normalizes them to our expected structure.
    It attempts multiple parsing strategies to handle different response patterns.
    
    Args:
        response_text: Raw response from the model
        
    Returns:
        Normalized dictionary with prediction and confidence fields
        
    Raises:
        ValueError: If response cannot be parsed or normalized
    """
    try:
        # Log raw response for debugging
        logging.debug(f"Raw model response: {response_text}")
        
        # Remove any markdown formatting
        clean_text = response_text.replace('```json', '').replace('```', '')
        
        # Try multiple parsing strategies
        parsed_response = None
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed_response = json.loads(clean_text.strip())
            logging.debug("Successfully parsed response as JSON")
        except json.JSONDecodeError:
            logging.debug("Failed to parse as direct JSON")
        
        # Strategy 2: Find JSON-like content
        if not parsed_response:
            import re
            json_pattern = r'\{[^}]+\}'
            matches = re.findall(json_pattern, clean_text)
            
            for match in matches:
                try:
                    parsed_response = json.loads(match)
                    logging.debug("Successfully extracted and parsed JSON content")
                    break
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Parse structured text response
        if not parsed_response:
            # Look for key phrases
            prediction_match = re.search(r'prediction[\s:"]*([YN]ES|NO)', clean_text, re.IGNORECASE)
            confidence_match = re.search(r'confidence[\s:"]*(\d+)', clean_text)
            
            if prediction_match and confidence_match:
                parsed_response = {
                    "prediction": prediction_match.group(1).upper(),
                    "confidence": int(confidence_match.group(1))
                }
                logging.debug("Successfully parsed structured text response")
        
        if not parsed_response:
            raise ValueError("Could not extract valid response from model output")
        
        # Normalize the response
        normalized = {}
        
        # Handle prediction
        pred_value = str(parsed_response.get('prediction', '')).upper()
        if pred_value in ['YES', 'NO']:
            normalized['prediction'] = pred_value
        else:
            # Try to interpret the prediction
            positive_indicators = ['YES', 'TRUE', '1', 'HIGH']
            if any(indicator in pred_value for indicator in positive_indicators):
                normalized['prediction'] = 'YES'
            else:
                normalized['prediction'] = 'NO'
        
        # Handle confidence
        conf_value = parsed_response.get('confidence', None)
        if conf_value is not None:
            if isinstance(conf_value, str):
                # Remove any % signs and convert to float
                conf_value = float(conf_value.replace('%', ''))
            normalized['confidence'] = int(min(max(float(conf_value), 0), 100))
        else:
            normalized['confidence'] = 90  # Default confidence if not provided
        
        logging.debug(f"Normalized response: {normalized}")
        return normalized
        
    except Exception as e:
        logging.error(f"Error processing model response: {str(e)}")
        logging.error(f"Raw response: {response_text}")
        raise

# Update the get_decision function to include a more explicit system message
async def get_decision(
    prompt_type: PromptType,
    model_name: str,
    config: Config,
    prompt: str
) -> Tuple[Optional[Decision], Optional[MetaData]]:
    """Get a decision from the model using the provided prompt"""
    try:
        system_message = (
            "You are a risk assessment expert. Your responses must be in valid JSON format "
            "containing exactly two fields: 'prediction' (either 'YES' or 'NO') and "
            "'confidence' (a number between 0 and 100). Do not include any additional text "
            "or explanations outside the JSON object."
        )
        
        # Make API call with configured parameters
        response = await asyncio.to_thread(
            chat,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            model=model_name,
            options={
                'temperature': config.model_config["model_temperature"],
                'top_p': config.model_config["model_top_p"],
                'max_tokens': config.model_config["model_max_tokens"],
            }
        )
        
        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            raise ValueError("Invalid API response structure")
        
        # Process and normalize the response
        normalized_response = process_model_response(response.message.content)
        
        # Create Decision object
        try:
            decision = Decision(
                prediction=normalized_response['prediction'],
                confidence=normalized_response['confidence']
            )
        except ValidationError as e:
            logging.error(f"Validation error creating Decision: {str(e)}")
            logging.error(f"Normalized response: {normalized_response}")
            return None, None
        
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
        
        return decision, meta_data
            
    except Exception as e:
        logging.error(f"Error during API call: {str(e)}")
        return None, None

async def get_decision_with_timeout(
    prompt_type: PromptType,
    model_name: str,
    config: Config,
    prompt: str
) -> Tuple[Optional[Decision], Optional[MetaData], TimeoutMetrics]:
    """
    Wraps get_decision in an asyncio timeout. If the model call times out,
    it returns (None, None, TimeoutMetrics) so the caller can handle it.
    """
    # Initialize timeout metrics; set default as no timeout occurrence
    timeout_metrics = TimeoutMetrics(
        occurred=False,
        retry_count=0
    )

    # Get desired timeout (in seconds) from config if available, or default to 30
    timeout_seconds = float(config.model_config.get("api_timeout", 30.0))

    try:
        # Attempt to get the decision within the specified timeout
        decision, meta_data = await asyncio.wait_for(
            get_decision(prompt_type, model_name, config, prompt),
            timeout=timeout_seconds
        )
        return decision, meta_data, timeout_metrics

    except asyncio.TimeoutError:
        logging.warning("API call timed out while waiting for model response.")
        # Mark that timeout happened
        timeout_metrics.occurred = True
        # Return None for decision, so the calling code knows it failed
        return None, None, timeout_metrics


def save_decision(
    decision: Decision,
    meta_data: MetaData,
    prompt_type: PromptType,
    model_name: str,
    row_id: int,
    actual_value: str,
    config: Config
) -> bool:
    """
    Save decision and metadata to filesystem.
    
    This function saves both the model's decision and associated metadata to a JSON
    file, including evaluation metrics like correctness of prediction.
    
    Args:
        decision: The model's decision
        meta_data: Associated metadata from the API response
        prompt_type: Type of prompt used
        model_name: Name of the model
        row_id: ID of the data row
        actual_value: Actual target value for evaluation
        config: Configuration settings
        
    Returns:
        bool indicating success of save operation
    """
    try:
        # Create output directory if needed
        output_dir = Path(config.output["base_dir"]) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with row ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{model_name}_{prompt_type}_id{row_id}_{timestamp}.json"
        
        # Prepare data for saving
        decision_data = pydantic_or_dict(decision)
        meta_data_data = pydantic_or_dict(meta_data)
        
        # Add evaluation metadata
        decision_data.update({
            'id': row_id,
            'actual': actual_value,
            'correct': "YES" if str(decision.prediction).upper() == actual_value.upper() else "NO"
        })
        
        # Convert timestamps in metadata
        meta_data_data = convert_ns_to_s(meta_data_data)
        
        # Combine all data
        combined_data = {
            "decision": decision_data,
            "meta_data": meta_data_data,
            "evaluation": {
                "timestamp": timestamp,
                "model": model_name,
                "prompt_type": str(prompt_type),
                "row_id": row_id,
                "prediction_matches_actual": decision_data['correct']
            }
        }
        
        # Save to file
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, default=str)
        
        logging.info(f"Successfully saved decision+meta_data to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {str(e)}")
        return False

def process_decision_response(response_text: str, prompt_type: PromptType) -> Optional[Decision]:
    """
    Process raw model response into a Decision object.
    
    This helper function handles parsing and validation of the model's response
    into a structured Decision object.
    
    Args:
        response_text: Raw response from the model
        prompt_type: Type of prompt used (affects expected response structure)
        
    Returns:
        Decision object if parsing successful, None otherwise
    """
    try:
        # Parse JSON response
        response_data = json.loads(response_text)
        
        # Validate and create Decision object
        decision = Decision.model_validate(response_data)
        
        # Log decision details
        logging.debug(
            f"Processed {prompt_type} decision: "
            f"{decision.prediction} ({decision.confidence}%)"
        )
        
        return decision
        
    except Exception as e:
        logging.error(f"Error processing decision response: {str(e)}")
        return None