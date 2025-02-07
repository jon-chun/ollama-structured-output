# config.py
import yaml
from pydantic import BaseModel, Field, ValidationError, conint, RootModel # Added RootModel import
from typing import Dict, Any, Optional, List

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
    delay_between_prompt_types_sec: int = Field(..., ge=0) # IMPORTANT: Field is here
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
    # Minimal load_config for testing - DOES NOT READ YAML, just returns a hardcoded Config
    model_ensemble_data = {
        "test_model": {"max_load_time": 120} # Minimal example model
    }
    config_data = {
        "model_parameters": {"api_type": "ollama", "api_model": "test_model", "model_temperature": 0.0, "model_top_p": 0.9, "model_max_tokens": 1000, "api_timeout": 60, "api_key_hardcoded": False, "openai_api_key": "", "anthropic_api_key": "", "google_api_key": ""},
        "api_parameters": {"api_delay": 0.1, "api_timeout": 30, "api_max_retries": 3, "rate_limits": {"openai": {"requests_per_minute": 60, "burst_size": 5, "retry_delay": 1.0, "max_retries": 3}, "anthropic": {"requests_per_minute": 30, "burst_size": 3, "retry_delay": 2.0, "max_retries": 3}, "google": {"requests_per_minute": 40, "burst_size": 4, "retry_delay": 1.5, "max_retries": 3}, "groq": {"requests_per_minute": 20, "burst_size": 2, "retry_delay": 2.5, "max_retries": 3}, "ollama": {"requests_per_minute": 50, "burst_size": 5, "retry_delay": 1.0, "max_retries": 3}}, "error_handling": {"backoff_multiplier": 2.0, "max_backoff": 60.0, "jitter": 0.1}},
        "execution": {"generation_type": "binary_classification", "max_calls_per_prompt": 1, "batch_size": 1, "nshot_ct": 3},
        "flags": {"max_samples": 10, "FLAG_PROMPT_PREFIX": False, "FLAG_PROMPT_SUFFIX": False, "prompt_prefix": "PREFIX", "prompt_suffix": "SUFFIX"},
        "timeout": {
            "max_wait_ollama_pull_model_sec": 300,
            "max_api_wait_sec": 60,
            "max_api_timeout_retries": 3,
            "api_wait_step_increase_sec": 30,
            "delay_between_prompt_types_sec": 1, # ADDED THIS LINE -  delay_between_prompt_types_sec
            "delay_between_model_load_sec": 1
        },
        "malformed_reponse": {"save_malformed_responses": False, "retry_malformed_ct": 3},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(levelname)s - %(message)s", "file": "test_config.log"},
        "output": {"base_dir": "test_output"},
        "data": {"input_file": "dummy_data.csv", "risk_factors": "text_column", "outcome": "label_column", "random_seed": 42, "train_split": 70, "validate_split": 15, "test_split": 15},
        "prompts": {"prompt_persona": "persona prompt", "system1": "system1 prompt", "cot": "cot prompt", "cot_nshot": "cot-nshot prompt"},
        "model_ensemble": model_ensemble_data # Use the minimal model_ensemble data
    }
    return Config(**config_data)