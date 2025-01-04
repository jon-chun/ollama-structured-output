# config.py
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

class Config:
    """
    Global configuration singleton for managing application settings.
    
    This class implements the singleton pattern to ensure only one configuration
    instance exists throughout the application. It loads settings from a YAML
    file and provides structured access to configuration parameters.
    
    Attributes:
        _instance: Singleton instance of the configuration
        _config: Loaded configuration dictionary
    """
    _instance = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls):
        """Ensure only one instance is created (singleton pattern)"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """
        Load configuration from config.yaml file.
        
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config.yaml is not found
            yaml.YAMLError: If config.yaml contains invalid YAML
        """
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required sections
            required_sections = [
                'model_config', 'execution', 'timeout', 'logging',
                'output', 'data', 'prompts', 'model_ensemble'
            ]
            
            missing_sections = [
                section for section in required_sections 
                if section not in config
            ]
            
            if missing_sections:
                raise ValueError(
                    f"Missing required configuration sections: {missing_sections}"
                )
                
            return config
            
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config.yaml: {str(e)}")
            raise

    @property
    def model_config(self) -> Dict[str, Any]:
        """Model configuration parameters"""
        return self._config["model_config"]

    @property
    def execution(self) -> Dict[str, Any]:
        """Execution settings"""
        return self._config["execution"]

    @property
    def timeout(self) -> Dict[str, Any]:
        """Timeout configuration"""
        return self._config["timeout"]

    @property
    def logging(self) -> Dict[str, Any]:
        """Logging configuration"""
        return self._config["logging"]

    @property
    def output(self) -> Dict[str, Any]:
        """Output directory configuration"""
        return self._config["output"]

    @property
    def data(self) -> Dict[str, Any]:
        """Data processing configuration"""
        return self._config["data"]

    @property
    def prompts(self) -> Dict[str, Any]:
        """Prompt templates"""
        return self._config["prompts"]

    @property
    def model_ensemble(self) -> Dict[str, Dict[str, Any]]:
        """Model ensemble configuration"""
        return self._config["model_ensemble"]

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model to get configuration for
            
        Returns:
            Model-specific configuration dictionary
            
        Raises:
            KeyError: If model_name is not found in configuration
        """
        if model_name not in self.model_ensemble:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        return self.model_ensemble[model_name]