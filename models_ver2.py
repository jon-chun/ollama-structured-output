# models.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Union, Optional, Any
from dataclasses import dataclass

class PromptType(str, Enum):
    """Enumeration of supported prompt types"""
    SYSTEM1 = 'system1'
    COT = 'cot'

class Prediction(str, Enum):
    """Enumeration for prediction outcomes"""
    YES = 'YES'
    NO = 'NO'

    @classmethod
    def normalize_prediction(cls, value: str) -> 'Prediction':
        """Normalize prediction value to enum"""
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid prediction: {value}. Must be one of {list(cls.__members__.keys())}.")

class RiskWeight(str, Enum):
    """Enumeration for risk weight levels"""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

    @classmethod
    def normalize_weight(cls, value: str) -> 'RiskWeight':
        """Normalize risk weight value to enum"""
        try:
            return cls[value.lower()]
        except KeyError:
            raise ValueError(f"Invalid risk weight: {value}. Must be one of {list(cls.__members__.keys())}.")

class RiskFactor(BaseModel):
    """Model for individual risk factors and their assessment"""
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=5)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionBase(BaseModel):
    """Base decision model with common fields"""
    prediction: Prediction
    confidence: int = Field(ge=0, le=100)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionCot(DecisionBase):
    """Extended decision model for chain-of-thought reasoning"""
    risk_factors: List[RiskFactor] = Field(..., min_items=1)

Decision = Union[DecisionBase, DecisionCot]