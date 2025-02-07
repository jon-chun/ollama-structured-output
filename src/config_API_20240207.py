# config.py
import yaml
from pydantic import RootModel, BaseModel, Field, ValidationError, conint # Re-confirm imports

from typing import Dict, Any, Optional

class RateLimitConfig(BaseModel): # Base class for rate limits
    requests_per_minute: int = Field(..., ge=1)
    burst_size: int = Field(..., ge=1)
    retry_delay: float = Field(..., ge=0)
    max_retries: int = Field(..., ge=0)

class OpenAIRateLimitConfig(RateLimitConfig):
    pass # Inherits from RateLimitConfig

class AnthropicRateLimitConfig(RateLimitConfig):
    pass # Inherits from RateLimitConfig

class GoogleRateLimitConfig(RateLimitConfig):
    pass # Inherits from RateLimitConfig

class GroqRateLimitConfig(RateLimitConfig):
    pass # Inherits from RateLimitConfig

class OllamaRateLimitConfig(RateLimitConfig):
    pass # Inherits from RateLimitConfig

class APIRateLimitsConfig(BaseModel):
    openai: OpenAIRateLimitConfig
    anthropic: AnthropicRateLimitConfig
    google: GoogleRateLimitConfig
    groq: GroqRateLimitConfig
    ollama: OllamaRateLimitConfig

class APIErrorHandlingConfig(BaseModel):
    backoff_multiplier: float = Field(..., ge=1.0)
    max_backoff: float = Field(..., ge=1.0)
    jitter: float = Field(..., ge=0.0, le=1.0)

class APIParametersConfig(BaseModel):
    api_delay: float = Field(..., ge=0.0)
    api_timeout: float = Field(..., ge=1.0)
    api_max_retries: int = Field(..., ge=0)
    rate_limits: APIRateLimitsConfig
    error_handling: APIErrorHandlingConfig

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

class TimeoutConfig(BaseModel): # Renamed Timeout to TimeoutConfig to avoid name clash if 'timeout' is used elsewhere
    max_wait_ollama_pull_model_sec: int = Field(..., ge=1)
    max_api_wait_sec: int = Field(..., ge=1)
    max_api_timeout_retries: int = Field(..., ge=0)
    api_wait_step_increase_sec: int = Field(..., ge=1)
    delay_between_prompt_types_sec: int = Field(..., ge=0)
    delay_between_model_load_sec: int = Field(..., ge=0)

class MalformedResponseConfig(BaseModel): # Renamed to MalformedResponseConfig
    save_malformed_responses: bool = Field(...)
    retry_malformed_ct: int = Field(..., ge=0)

class LoggingConfig(BaseModel): # Renamed to LoggingConfig
    level: str = Field(...)
    format: str = Field(...)
    file: str = Field(...)

class OutputConfig(BaseModel): # Renamed to OutputConfig
    base_dir: str = Field(...)

class DataConfig(BaseModel): # Renamed to DataConfig
    input_file: str = Field(...)
    risk_factors: str = Field(...)
    outcome: str = Field(...)
    random_seed: int = Field(...)
    train_split: conint(ge=0, le=100) = Field(...)
    validate_split: conint(ge=0, le=100) = Field(...)
    test_split: conint(ge=0, le=100) = Field(...)

class PromptsConfig(BaseModel): # Renamed to PromptsConfig
    prompt_persona: str = Field(...)
    system1: str = Field(...)
    cot: str = Field(...)
    cot_nshot: str = Field(...)


class ModelEnsembleConfig(RootModel[Dict[str, Dict[str, Any]]]): # Values are Dict[str, Any] now
    pass

class ModelParametersConfig(BaseModel): # Renamed to ModelParametersConfig
    api_type: str = Field(...)
    api_model: str = Field(...)
    model_temperature: float = Field(...)
    model_top_p: float = Field(...)
    model_max_tokens: int = Field(...)
    api_timeout: int = Field(...) # Might remove this as api_parameters.api_timeout is more central
    api_key_hardcoded: bool = Field(...)
    openai_api_key: str = Field(...)
    anthropic_api_key: str = Field(...)
    google_api_key: str = Field(...)

class Config(BaseModel):
    model_parameters: ModelParametersConfig # Use ModelParametersConfig model
    api_parameters: APIParametersConfig # NEW: API parameters config
    execution: ExecutionConfig
    flags: FlagsConfig
    timeout: TimeoutConfig # Use TimeoutConfig model
    malformed_reponse: MalformedResponseConfig # Use MalformedResponseConfig
    logging: LoggingConfig # Use LoggingConfig model
    output: OutputConfig # Use OutputConfig model
    data: DataConfig # Use DataConfig model
    prompts: PromptsConfig # Use PromptsConfig model
    model_ensemble: ModelEnsembleConfig # Use ModelEnsembleConfig

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