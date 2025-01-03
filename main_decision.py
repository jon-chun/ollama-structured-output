from ollama import chat
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import List, Optional
import logging

MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.9
MODEL_MAX_TOKENS = 1024

PROMPT_TYPE = '' # Defined here, but set in main

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = 'llama3.1:8b-instruct-q4_K_M' # 'Phi-4:Q4_K_M'  # Replace with actual model if necessary

# Prompt texts
PROMPT_SYSTEM1 = """
###TASK: 
Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

###RISK_FACTORS:
The person is a 22-year-old male who identifies himself as White. He has been arrested 2 times and has been convicted. He is 5 feet 6 inches and weighs 120 lbs. He has never been married and is currently living alone. He does not have a high school diploma or GED certificate and is not enrolled in college. He has held 3 jobs during the year. He has used drugs in the past four years (marijuana and cocaine) and some of the time during the past month he has felt so down in the dumps that nothing could cheer him up. At a young age, he lived with his biological dad in a household of size 4, in a rural area. He considers that God has nothing to do with what happens to him personally.

###RESPONSE_FORMAT:
1. 'YES' or 'NO' (prediction)
2. integer 0-100 (confidence in percent)
"""

PROMPT_COT = """
###TASK:
Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

###RISK_FACTORS:
The person is an 18-year-old male who identifies himself as White. He has been arrested 10 times and has been convicted. He is 6 feet 1 inch and weighs 195 lbs. He has never been married and is currently living alone. He has a GED certificate and is not enrolled in college. He has held 1 job during the year. He has used drugs in the past four years (marijuana and cocaine) and has not felt depressed over the past month. At a young age, he lived with both of his biological parents in a household of size 4, in an urban area. He considers that God has something to do with what happens to him personally.

###RESPONSE_FORMAT:
1. 'YES' or 'NO' (prediction)
2. integer 0-100 (confidence in percent)
3. A list of text strings identifying each risk factor with reasoning to weight each risk factor as 'high', 'medium' or 'low'
"""

# Enum for Prompt Types
class PromptType(str, Enum):
    SYSTEM1 = 'system1'
    COT = 'cot'

# Enum for Predictions
class Prediction(str, Enum):
    YES = 'YES'
    NO = 'NO'

    @classmethod
    def normalize_prediction(cls, value: str) -> 'Prediction':
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid prediction value: {value}. Must be one of {list(cls.__members__.keys())}."
            )

# Enum for Risk Weights
class RiskWeight(str, Enum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

    @classmethod
    def normalize_weight(cls, value: str) -> 'RiskWeight':
        try:
            return cls[value.lower()]
        except KeyError:
            raise ValueError(
                f"Invalid risk weight: {value}. Must be one of {list(cls.__members__.keys())}."
            )

# Pydantic Model for Risk Factors
class RiskFactor(BaseModel):
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=5)

    class Config:
        frozen = True
        extra = 'forbid'

# Decision Models
class DecisionSystem1(BaseModel):
    prediction: Prediction
    confidence: int = Field(ge=0, le=100)

    class Config:
        frozen = True
        extra = 'forbid'

class DecisionCot(DecisionSystem1):
    risk_factors: List[RiskFactor] = Field(..., min_items=1)

# Response Processor
class ResponseProcessor:
    def __init__(self, prompt_type: PromptType):
        self.prompt_type = prompt_type
        self.decision_class = DecisionSystem1 if prompt_type == PromptType.SYSTEM1 else DecisionCot

    def process_response(self, response_content: str) -> Optional[BaseModel]:
        try:
            decision = self.decision_class.model_validate_json(response_content)
            self._log_decision(decision)
            return decision
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing response: {e}")
            return None

    def _log_decision(self, decision: BaseModel) -> None:
        logger.info(f"Prediction: {decision.prediction}")
        logger.info(f"Confidence: {decision.confidence}")

        if isinstance(decision, DecisionCot):
            for rf in decision.risk_factors:
                logger.info(f"Risk Factor: {rf.factor} ({rf.weight}): {rf.reasoning}")

# Main Function
def get_decision(prompt_type: PromptType, model: str = OLLAMA_MODEL) -> Optional[BaseModel]:
    processor = ResponseProcessor(prompt_type)
    prompt_str = PROMPT_SYSTEM1 if prompt_type == PromptType.SYSTEM1 else PROMPT_COT

    try:
        response = chat(
            messages=[{'role': 'user', 'content': prompt_str}],
            model=model,
            options={
                'temperature': MODEL_TEMPERATURE,
                'top_p': MODEL_TOP_P,
                'max_tokens': MODEL_MAX_TOKENS,
            },
            format=processor.decision_class.model_json_schema(),
        )
        if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
            logger.error("Invalid API response structure.")
            return None

        return processor.process_response(response.message.content)
    except Exception as e:
        logger.error(f"Error during API call or processing: {e}")
        return None

# Entry Point
if __name__ == "__main__":
    PROMPT_TYPE = PromptType.COT
    # PROMPT_TYPE = PromptType.SYSTEM1
    
    try:
        decision = get_decision(PROMPT_TYPE)
        if decision:
            print(f"\nPREDICTION: {decision.prediction}")
            print(f"CONFIDENCE: {decision.confidence}")

            if isinstance(decision, DecisionCot):
                print("\nRISK FACTORS:")
                for rf in decision.risk_factors:
                    print(f"- {rf.factor} ({rf.weight.value}): {rf.reasoning}")
    except Exception as e:
        logger.error(f"Application error: {e}")
