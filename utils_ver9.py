# utils.py
import re
from pathlib import Path
from models import PromptType
from config import Config
import json
from pydantic import BaseModel
from typing import Dict, Any, Optional, Set, DefaultDict, Tuple, TypeAlias, Literal, Union
from collections import defaultdict
import logging

PromptTypeStr: TypeAlias = Literal['system1', 'cot', 'cot-nshot']

VALID_PROMPT_TYPES: Set[str] = {
    'system1',
    'cot',
    'cot-nshot'
}


def get_prompt_type_str(prompt_type: Union[PromptType, str]) -> str:
    """
    Convert a PromptType enum or string to its corresponding config key string.
    Handles both enum values and string representations.
    
    Args:
        prompt_type: Either a PromptType enum value or string representation
        
    Returns:
        Corresponding prompt type string from VALID_PROMPT_TYPES
        
    Raises:
        ValueError: If the prompt type is invalid or cannot be converted
    """
    try:
        if isinstance(prompt_type, PromptType):
            result = prompt_type.value
        elif isinstance(prompt_type, str):
            if prompt_type.startswith('PromptType.'):
                # Handle string representation of enum (e.g., "PromptType.SYSTEM1")
                enum_key = prompt_type.split('.')[-1]
                result = getattr(PromptType, enum_key).value
            else:
                # Handle direct string values (e.g., "system1")
                result = prompt_type
        else:
            raise ValueError(f"Invalid prompt type format: {prompt_type}")
            
        # Validate result
        if result not in VALID_PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type value: {result}")
            
        return result
        
    except Exception as e:
        logging.error(f"Error converting prompt type {prompt_type}: {str(e)}")
        raise


def check_model_prompt_completion(output_dir: Path, model_name: str, prompt_type: Union[PromptType, str], 
                                max_samples: int, max_calls_per_prompt: int) -> bool:
    """
    Check if a model-prompt combination has reached its maximum allowed outputs.
    Uses the correct nested directory structure based on prompt type.
    """
    try:
        prompt_type_str = get_prompt_type_str(prompt_type)
        clean_name = clean_model_name(model_name)
        model_dir = output_dir / clean_name / prompt_type_str
        
        if not model_dir.exists():
            return False
            
        # Get unique filename roots by removing datetime suffix
        unique_roots = set()
        for file_path in model_dir.glob(f"{model_name}_{prompt_type_str}_*.json"):
            base_name = file_path.stem
            root = '_'.join(base_name.split('_')[:-1])  # Remove datetime part
            unique_roots.add(root)
        
        max_allowed = max_samples * max_calls_per_prompt
        return len(unique_roots) >= max_allowed
        
    except Exception as e:
        logging.error(f"Error in check_model_prompt_completion: {str(e)}")
        raise

def check_existing_output(output_dir: Path, model_name: str, prompt_type: Union[PromptType, str], 
                         row_id: int) -> bool:
    """
    Check if output file already exists for this model-prompt-id combination.
    Uses the correct nested directory structure based on prompt type.
    """
    try:
        prompt_type_str = get_prompt_type_str(prompt_type)
        clean_name = clean_model_name(model_name)
        model_dir = output_dir / clean_name / prompt_type_str
        
        if not model_dir.exists():
            return False
        
        pattern = f"{model_name}_{prompt_type_str}_id{row_id}_*.json"
        matching_files = list(model_dir.glob(pattern))
        
        return len(matching_files) > 0
        
    except Exception as e:
        logging.error(f"Error in check_existing_output: {str(e)}")
        raise

def is_combination_fully_complete(
    completion_counts: Dict[int, int],
    max_samples: int,
    max_calls: int
) -> bool:
    """
    Check if we have enough samples with enough calls each.
    """
    fully_completed = sum(1 for count in completion_counts.values() 
                         if count >= max_calls)
    return fully_completed >= max_samples

def get_completion_counts(output_dir: Path, model_name: str, prompt_type: PromptTypeStr) -> Dict[int, int]:
    """
    Get count of completions for each row ID for a specific model and prompt type.
    Returns a dictionary mapping row_id to count of completions.
    """
    clean_name = clean_model_name(model_name)
    model_dir = output_dir / clean_name / prompt_type
    
    if not model_dir.exists():
        return {}
        
    completion_counts = {}
    
    # Scan all output files for this model+prompt combination
    for file_path in model_dir.glob(f"{model_name}_{prompt_type}_id*_*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                row_id = data['decision']['id']
                completion_counts[row_id] = completion_counts.get(row_id, 0) + 1
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Error reading file {file_path}: {e}")
            continue
            
    return completion_counts

def get_completion_status(
    output_dir: Path,
    model_name: str,
    prompt_type: PromptTypeStr,
    max_samples: int,
    max_calls: int
) -> Tuple[bool, Dict[int, int]]:
    """
    Check if a model/prompt combination has completed its required samples and calls.
    Returns (is_complete, completed_calls_dict)
    """
    completion_counts = get_completion_counts(output_dir, model_name, prompt_type)
    is_complete = is_combination_fully_complete(completion_counts, max_samples, max_calls)
    
    logging.info(
        f"Status for {model_name} - {prompt_type}: "
        f"completed_samples={sum(1 for calls in completion_counts.values() if calls >= max_calls)}, "
        f"max_samples={max_samples}, "
        f"is_complete={is_complete}"
    )
    
    return is_complete, completion_counts

def clean_model_name(model_name: str) -> str:
    """
    Convert model name to OS-friendly format:
    - Collapse contiguous whitespace to single underscore
    - Convert colons to underscores
    - Convert other punctuation to hyphens
    - Convert to lowercase
    """
    cleaned = re.sub(r'\s+', '_', model_name.strip().lower())
    cleaned = cleaned.replace(':', '_')
    cleaned = re.sub(r'[^\w_]', '-', cleaned)
    return cleaned

def get_next_sample(
    completion_counts: Dict[int, int],
    max_calls: int,
    row_id: int
) -> int:
    """
    Get the next needed repeat index for a sample.
    
    Args:
        completion_counts: Dictionary mapping row_id to number of completions
        max_calls: Maximum number of calls allowed per sample
        row_id: The ID of the row to check
        
    Returns:
        The next needed repeat index, or None if sample is complete
        
    Example:
        If max_calls=3 and row_id has 2 completions, returns 2
        If max_calls=3 and row_id has 3 completions, returns None
    """
    current_count = completion_counts.get(row_id, 0)
    if current_count >= max_calls:
        return None
    return current_count

def check_existing_decision(
    model_name: str,
    prompt_type: PromptTypeStr,
    row_id: int,
    repeat_index: int,
    config: Any,
    nshot_ct: Optional[int] = None
) -> bool:
    """Check if a specific decision file exists."""
    clean_name = clean_model_name(model_name)
    output_dir = Path(config.output["base_dir"]) / clean_name
    
    if not output_dir.exists():
        return False
    
    if prompt_type == "COT_NSHOT":
        pattern = f"{model_name}_{prompt_type}_id{row_id}_nshot{nshot_ct}_*.json"
    else:
        pattern = f"{model_name}_{prompt_type}_id{row_id}_ver{repeat_index}_*.json"
    
    existing_files = list(output_dir.glob(pattern))
    return len(existing_files) > 0

def pydantic_or_dict(obj: Any) -> Dict:
    """Convert pydantic model to dict if needed"""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj

def convert_ns_to_s(data: Dict) -> Dict:
    """Convert nanosecond values to seconds in metadata"""
    ns_keys = [
        'total_duration', 'load_duration', 
        'prompt_eval_duration', 'eval_duration'
    ]
    for key in ns_keys:
        if key in data and data[key] is not None:
            data[key] = data[key] / 1e9
    return data

def count_unique_samples(output_dir: Path, model_name: str, prompt_type: str) -> Set[int]:
    """
    Count unique sample IDs that already have output files.
    
    Args:
        output_dir: Base output directory
        model_name: Name of the model being evaluated
        prompt_type: Type of prompt being used
        
    Returns:
        Set of unique sample IDs that already have outputs
    """
    clean_name = clean_model_name(model_name)
    prompt_dir = output_dir / clean_name / prompt_type
    
    if not prompt_dir.exists():
        return set()
        
    unique_ids = set()
    for file_path in prompt_dir.glob(f"{model_name}_{prompt_type}_id*_*.json"):
        try:
            # Extract ID from filename using regex
            match = re.search(r'_id(\d+)_', file_path.name)
            if match:
                unique_ids.add(int(match.group(1)))
        except Exception as e:
            logging.warning(f"Error extracting ID from {file_path}: {e}")
            
    return unique_ids