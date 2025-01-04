# utils.py
from typing import Any, Dict, Union
import json

def pydantic_or_dict(obj: Any) -> Dict:
    """
    Convert a Pydantic model to dict or return the object if it's already dict-like.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary representation of the object
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj

def convert_ns_to_s(data: Any) -> Any:
    """
    Recursively convert nanosecond timestamps to seconds.
    
    Args:
        data: Data structure potentially containing nanosecond timestamps
        
    Returns:
        Data structure with timestamps converted to seconds
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                convert_ns_to_s(value)
            elif isinstance(value, (int, float)) and value > 1e7:
                data[key] = value / 1e9
    
    elif isinstance(data, list):
        for i, value in enumerate(data):
            if isinstance(value, (dict, list)):
                convert_ns_to_s(value)
            elif isinstance(value, (int, float)) and value > 1e7:
                data[i] = value / 1e9
    
    return data

def load_json(file_path: str) -> Dict:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Path where to save the file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)