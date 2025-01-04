# llm_decisions/models.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .config import Prediction, RiskWeight  # Use relative import within package

# Base Models
class RiskFactor(BaseModel):
    """Model for individual risk factors"""
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=5)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionBase(BaseModel):
    """Base decision model"""
    prediction: Prediction
    confidence: int = Field(ge=0, le=100)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionCot(DecisionBase):
    """Chain of thought decision model"""
    risk_factors: List[RiskFactor] = Field(..., min_items=1)

# Metrics Models
@dataclass
class TimeoutMetrics:
    """Timeout tracking metrics"""
    occurred: bool
    retry_count: int = 0
    final_timeout_duration: Optional[float] = None
    total_timeout_duration: float = 0

@dataclass
class ApiCallMetadata:
    """API call performance metrics"""
    total_duration: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_token_time: float
    tokens_per_second: float
    raw_response: Dict[str, Any]
    
    def __post_init__(self):
        """Validate metrics"""
        if self.total_duration < 0:
            raise ValueError("Duration cannot be negative")
        if self.tokens_per_second < 0:
            raise ValueError("Tokens per second cannot be negative")

@dataclass
class PromptMetrics:
    """Prompt execution metrics"""
    attempt_number: int
    execution_time_seconds: float
    successful: bool
    timeout_metrics: TimeoutMetrics
    error_message: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        """Convert types and validate"""
        self.execution_time_seconds = float(self.execution_time_seconds)
        if self.prediction is not None:
            self.prediction = str(self.prediction)
        if self.confidence is not None:
            self.confidence = float(self.confidence)

@dataclass
class TimeoutStats:
    """Aggregated timeout statistics"""
    total_timeouts: int
    avg_timeout_duration: float
    max_timeout_duration: float
    total_timeout_duration: float

@dataclass
class AggregateApiMetadata:
    """Aggregated API performance metrics"""
    total_calls: int
    total_duration: float
    avg_duration: float
    total_prompt_tokens: int
    avg_prompt_tokens: float
    total_completion_tokens: int
    avg_completion_tokens: float
    total_tokens: int
    avg_tokens: float
    avg_first_token_time: float
    avg_tokens_per_second: float
    token_distribution: Dict[str, Dict[str, float]]

@dataclass
class PerformanceStats:
    """Overall performance statistics"""
    prompt_type: str
    model_name: str
    start_time: datetime
    end_time: datetime
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    timeout_attempts: int
    avg_execution_time: float
    median_execution_time: float
    timeout_stats: TimeoutStats
    api_metrics: AggregateApiMetadata