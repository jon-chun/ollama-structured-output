# models.py
from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class PromptType(str, Enum):
    """Enumeration of supported prompt types"""
    SYSTEM1 = 'system1'
    COT = 'cot'
    COT_NSHOT = 'cot-nshot'  # New prompt type for n-shot learning

class RiskWeight(str, Enum):
    """Enumeration for risk weight levels"""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class RiskFactor(BaseModel):
    """Model for individual risk factors and their assessment"""
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=5)

class Decision(BaseModel):
    """Base decision model with prediction and confidence"""
    prediction: str = Field(
        ...,
        description="The model's prediction (YES/NO)",
        pattern="^(YES|NO)$"
    )
    confidence: int = Field(
        ...,
        description="Confidence level (0-100)",
        ge=0,
        le=100
    )

class DecisionWithRiskFactors(Decision):
    """Extended decision model with risk factor analysis"""
    risk_factors: List[RiskFactor] = Field(
        ...,
        description="Analyzed risk factors with weights and reasoning"
    )