# config.py

import yaml
from pydantic import BaseModel
from typing import Dict, Any

class Config(BaseModel):
    model_parameters: Dict[str, Any]
    execution: Dict[str, Any]
    flags: Dict[str, Any]
    timeout: Dict[str, Any]
    logging: Dict[str, Any]
    output: Dict[str, Any]
    data: Dict[str, Any]
    prompts: Dict[str, str]
    model_ensemble: Dict[str, Dict[str, Any]]

    @property
    def max_samples(self) -> int:
        """
        Return the maximum number of samples specified in the config.
        If <= 0, then we do not limit the samples.
        """
        return int(self.flags.get("max_samples", 0))

def load_config(config_file: str = "config.yaml") -> Config:
    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return Config(**yaml_data)
