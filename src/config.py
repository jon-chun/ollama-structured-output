import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any

class ExecutionConfig(BaseModel):
    max_calls_per_prompt: int = Field(..., ge=1, description="Maximum number of calls per prompt/sample")
    batch_size: int = Field(..., ge=1, description="Number of samples per batch")
    nshot_ct: int = Field(..., ge=0, description="Number of n-shot examples to include")

class FlagsConfig(BaseModel):
    max_samples: int = Field(..., ge=1, description="Maximum number of samples to process")
    FLAG_PROMPT_PREFIX: bool = Field(..., description="Enable prompt prefix")
    FLAG_PROMPT_SUFFIX: bool = Field(..., description="Enable prompt suffix")
    prompt_prefix: str = Field(..., description="Prefix to prepend to prompts")
    prompt_suffix: str = Field(..., description="Suffix to append to prompts")

class Config(BaseModel):
    model_parameters: Dict[str, Any]
    execution: ExecutionConfig
    flags: FlagsConfig
    timeout: Dict[str, Any]
    logging: Dict[str, Any]
    output: Dict[str, Any]
    data: Dict[str, Any]
    prompts: Dict[str, str]
    model_ensemble: Dict[str, Dict[str, Any]]

    @property
    def max_samples(self) -> int:
        return self.flags.max_samples

    @property
    def max_calls_per_prompt(self) -> int:
        return self.execution.max_calls_per_prompt

    @property
    def batch_size(self) -> int:
        return self.execution.batch_size

    @property
    def nshot_ct(self) -> int:
        return self.execution.nshot_ct

def load_config(config_file: str = "config.yaml") -> Config:
    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    try:
        return Config(**yaml_data)
    except ValidationError as ve:
        print(f"Configuration validation error: {ve}")
        raise