# Standard library imports
from pathlib import Path
from typing import Dict, Any, Final, Optional
from enum import Enum
import yaml
import logging

# Constants
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BACKOFF_FACTOR: Final[float] = 1.5

# Enums for type safety
class PromptType(str, Enum):
    """Types of prompts supported by the system"""
    SYSTEM = 'system'
    COT = 'cot'  # Chain of Thought

class Prediction(str, Enum):
    """Possible prediction outcomes"""
    YES = 'YES'
    NO = 'NO'
    
    @classmethod
    def normalize(cls, value: str) -> 'Prediction':
        """Normalize prediction values to enum"""
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid prediction: {value}")

class RiskWeight(str, Enum):
    """Risk assessment weights"""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    
    @classmethod
    def normalize(cls, value: str) -> 'RiskWeight':
        """Normalize risk weight values to enum"""
        try:
            return cls[value.lower()]
        except KeyError:
            raise ValueError(f"Invalid risk weight: {value}")

# Configuration Management
class ConfigurationError(Exception):
    """Raised when configuration validation fails"""
    pass

class Config:
    """Global configuration with validation and type safety"""
    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or Path("config.yaml")
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse configuration file"""
        if not self._config_path.exists():
            raise ConfigurationError(f"Config file not found: {self._config_path}")
        
        try:
            with open(self._config_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config: {e}")
    
    def _validate_config(self):
        """Validate configuration values"""
        required_sections = {
            "model_config", "execution", "timeout", 
            "logging", "output", "model_ensemble"
        }
        
        if missing := required_sections - self._config.keys():
            raise ConfigurationError(f"Missing config sections: {missing}")
        
        self._validate_model_config()
        self._validate_timeout_config()
    
    def _validate_model_config(self):
        """Validate model-specific configuration"""
        required_keys = {
            "model_temperature", "model_top_p", 
            "model_max_tokens"
        }
        
        config = self._config.get("model_config", {})
        if missing := required_keys - config.keys():
            raise ConfigurationError(f"Missing model config keys: {missing}")
    
    def _validate_timeout_config(self):
        """Validate timeout configuration"""
        timeout_config = self._config.get("timeout", {})
        required_keys = {
            "max_api_wait_sec", 
            "max_api_timeout_retries",
            "api_wait_step_increase_sec"
        }
        
        if missing := required_keys - timeout_config.keys():
            raise ConfigurationError(f"Missing timeout config keys: {missing}")
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration settings"""
        return self._config["model_config"]
    
    @property
    def execution(self) -> Dict[str, Any]:
        """Get execution settings"""
        return self._config["execution"]
    
    @property
    def timeout(self) -> Dict[str, Any]:
        """Get timeout settings"""
        return self._config["timeout"]
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._config["logging"]
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output settings"""
        return self._config["output"]
    
    @property
    def model_ensemble(self) -> Dict[str, Dict[str, Any]]:
        """Get model ensemble configuration"""
        return self._config["model_ensemble"]