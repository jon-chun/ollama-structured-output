# utils.py
import re
from pathlib import Path
from models import PromptType
import json
from pydantic import BaseModel
from typing import Dict, Any, Optional, Set, DefaultDict, Tuple
from collections import defaultdict
import logging

# OLD: No existing function

# NEW:
def check_model_prompt_completion(output_dir: Path, model_name: str, prompt_type: str, 
                                max_samples: int, max_calls_per_prompt: int) -> bool:
    """
    Check if a model-prompt combination has reached its maximum allowed outputs.
    Returns True if max outputs reached, False otherwise.
    """
    clean_name = clean_model_name(model_name)
    model_dir = output_dir / clean_name
    
    if not model_dir.exists():
        return False
        
    # Get unique filename roots by removing datetime suffix
    unique_roots = set()
    for file_path in model_dir.glob(f"{model_name}_{prompt_type}_*.json"):
        # Extract base filename without datetime suffix
        base_name = file_path.stem
        root = '_'.join(base_name.split('_')[:-1])  # Remove datetime part
        unique_roots.add(root)
    
    max_allowed = max_samples * max_calls_per_prompt
    return len(unique_roots) >= max_allowed

def get_completion_counts(
    output_dir: Path,
    model_name: str,
    prompt_type: str
) -> dict[int, int]:
    """
    Get counts of successful completions for each sample ID.
    
    Returns:
        Dict[int, int]: Map of sample_id -> number of successful completions
    """
    clean_name = clean_model_name(model_name)
    model_dir = output_dir / clean_name
    
    if not model_dir.exists():
        return {}
        
    completion_counts = defaultdict(int)
    file_pattern = f"{model_name}_{prompt_type}_id*_*.json"
    
    logging.debug(f"Checking files matching {file_pattern} in {model_dir}")
    for file_path in model_dir.glob(file_pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Only count successful, valid outputs
                if data['decision'].get('prediction') and data['decision'].get('confidence'):
                    row_id = data['decision']['id']
                    completion_counts[row_id] += 1
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {e}")
            continue
            
    return dict(completion_counts)

def is_combination_fully_complete(
    counts: Dict[int, int],
    max_samples: int,
    max_calls: int
) -> bool:
    """
    Check if we have enough complete samples with required calls.
    """
    completed_samples = sum(1 for calls in counts.values() if calls >= max_calls)
    return completed_samples >= max_samples

def get_completion_status(
    output_dir: Path,
    model_name: str,
    prompt_type: str,
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
    existing_calls: Dict[int, int],
    max_calls: int,
    row_id: int
) -> Optional[int]:
    """
    Determine the next repeat index needed for a sample, if any.
    """
    existing_count = existing_calls.get(row_id, 0)
    if existing_count >= max_calls:
        return None
    return existing_count

def check_existing_decision(
    model_name: str,
    prompt_type: str,
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