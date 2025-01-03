from ollama import chat
from pydantic import BaseModel, Field, ValidationError
from enum import Enum, auto
from typing import List, Optional
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = 'Phi-4:Q4_K_M' # 'Phi-4:Q4_K_M' or 'llama3.2:3b', 

prompt_system1 = """
###TASK: 
Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

###RISK_FACTORS:
The person is a 22-year-old male who identifies himself as White. He has been arrested 2 times and has been convicted. He is 5 feet 6 inches and weighs 120 lbs. He has never been married and is currently living alone. He does not have a high school diploma or GED certificate and is not enrolled in college. He has held 3 jobs during the year. He has used drugs in the past four years (marijuana and cocaine) and some of the time during the past month he has felt so down in the dumps that nothing could cheer him up. At a young age, he lived with his biological dad in a household of size 4, in a rural area. He considers that God has nothing to do with what happens to him personally.

###RESPONSE_FORMAT:
1. 'YES' or 'NO' (prediction)
2. integer 0-100 (confidence in percent)


###INSTRUCTIONS:
Please response with only a one word “YES” or “NO” and an integer 0-100 confidence percentage for your 3 year recidivism prediction. DO NOT say 'Hello' or any greetings. Use your judicial and statistical expertise with these ###RISK_FACTORS to predict if the person will be rearrested within the next 3 years. Do not provide any description or reasoning. ONLY respond with “YES” or “NO” for your 3 year recidivism prediction. Return your responses in the form of ###RESPONSE_FORMAT. DO NOT say 'Hello' or any other greetings.
"""

prompt_cot = """
###TASK:
Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

###RISK_FACTORS:
The person is an 18-year-old male who identifies himself as White. He has been arrested 10 times and has been convicted. He is 6 feet 1 inch and weighs 195 lbs. He has never been married and is currently living alone. He has a GED certificate and is not enrolled in college. He has held 1 job during the year. He has used drugs in the past four years (marijuana and cocaine) and has not felt depressed over the past month. At a young age, he lived with both of his biological parents in a household of size 4, in an urban area. He considers that God has something to do with what happens to him personally.

###RESPONSE_FORMAT:
1. 'YES' or 'NO' (prediction)
2. integer 0-100 (confidence in percent)
3. a list of text strings identifying each risk factor with reasoning to weight each risk factor as 'high', 'medium' or 'low'


list of text strings with reasoning weighting each risk factor 
'YES' or 'NO' (prediction)
integer 0-100 (confidence in percent)

###INSTRUCTIONS:
Please response with only a one word “YES” or “NO”, an integer 0-100 confidence percentage, and a list of text strings with reasoning weighting each risk factor for your 3 year recidivism prediction. DO NOT say 'Hello' or any greetings. Use your judicial and statistical expertise with these ###RISK_FACTORS to predict if the person will be rearrested within the next 3 years. Do not provide any description or reasoning. ONLY respond with “YES” or “NO” for your 3 year recidivism prediction. Return your responses in the form of ###RESPONSE_FORMAT. DO NOT say 'Hello' or any other greetings.
"""





class PromptType(str, Enum):
    """Defines the type of prompt being used to ensure type safety"""
    PROMPT_SYSTEM1 = '''
    ###TASK:
    Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

    ###RISK_FACTORS:
    The person is an 18-year-old male who identifies himself as White. He has been arrested 10 times and has been convicted. He is 6 feet 1 inch and weighs 195 lbs. He has never been married and is currently living alone. He has a GED certificate and is not enrolled in college. He has held 1 job during the year. He has used drugs in the past four years (marijuana and cocaine) and has not felt depressed over the past month. At a young age, he lived with both of his biological parents in a household of size 4, in an urban area. He considers that God has something to do with what happens to him personally.

    ###RESPONSE_FORMAT:
    1. 'YES' or 'NO' (prediction)
    2. integer 0-100 (confidence in percent)

    list of text strings with reasoning weighting each risk factor 
    'YES' or 'NO' (prediction)
    integer 0-100 (confidence in percent)

    ###INSTRUCTIONS:
    Please response with only a one word “YES” or “NO” and an integer 0-100 confidence percentage. Return your responses in the form of ###RESPONSE_FORMAT. DO NOT say 'Hello' or any other greetings.
    '''

    PROMPT_COT = '''
    ###TASK:
    Your task is to predict if a person will be rearrested within 3 years given (1) your expertise, and (2) the given ###RISK_FACTORS in this case which may be predictors of varying strengths

    ###RISK_FACTORS:
    The person is an 18-year-old male who identifies himself as White. He has been arrested 10 times and has been convicted. He is 6 feet 1 inch and weighs 195 lbs. He has never been married and is currently living alone. He has a GED certificate and is not enrolled in college. He has held 1 job during the year. He has used drugs in the past four years (marijuana and cocaine) and has not felt depressed over the past month. At a young age, he lived with both of his biological parents in a household of size 4, in an urban area. He considers that God has something to do with what happens to him personally.

    ###RESPONSE_FORMAT:
    1. 'YES' or 'NO' (prediction)
    2. integer 0-100 (confidence in percent)
    3. a list of text strings identifying each risk factor with reasoning to weight each risk factor as 'high', 'medium' or 'low'

    ###INSTRUCTIONS:
    Please response with only a one word “YES” or “NO”, an integer 0-100 confidence percentage, and a list of text strings with reasoning weighting each risk factor for your 3 year recidivism prediction. DO NOT say 'Hello' or any greetings. Use your judicial and statistical expertise with these ###RISK_FACTORS to predict if the person will be rearrested within the next 3 years. Do not provide any description or reasoning. ONLY respond with “YES” or “NO” for your 3 year recidivism prediction. Return your responses in the form of ###RESPONSE_FORMAT. DO NOT say 'Hello' or any other greetings.
    '''

class Prediction(str, Enum):
    """Defines possible prediction outcomes with case normalization"""
    YES = 'YES'
    NO = 'NO'

    @classmethod
    def normalize_prediction(cls, value: str) -> 'Prediction':
        """
        Normalizes prediction input with robust error handling
        
        Args:
            value: Input string to normalize
            
        Returns:
            Normalized Prediction enum
            
        Raises:
            ValueError: If input cannot be normalized to valid prediction
        """
        try:
            return cls[value.upper()]
        except KeyError:
            logger.error(f"Invalid prediction value: {value}")
            raise ValueError(f"Prediction must be one of {list(cls.__members__.keys())}, got {value}")

class RiskWeight(str, Enum):
    """Defines risk weight levels with case normalization"""
    HIGH = 'high'
    MEDIUM = 'medium' 
    LOW = 'low'

    @classmethod
    def normalize_weight(cls, value: str) -> 'RiskWeight':
        """
        Normalizes risk weight input with robust error handling
        
        Args:
            value: Input string to normalize
            
        Returns:
            Normalized RiskWeight enum
            
        Raises:
            ValueError: If input cannot be normalized to valid weight
        """
        try:
            return cls[value.lower()]
        except KeyError:
            logger.error(f"Invalid risk weight: {value}")
            raise ValueError(f"Risk weight must be one of {list(cls.__members__.keys())}, got {value}")
        



class RiskFactor(BaseModel):
    """Structured representation of a risk factor assessment"""
    factor: str = Field(..., min_length=1)
    weight: RiskWeight
    reasoning: str = Field(..., min_length=10)  # Ensure meaningful reasoning
    
    class Config:
        """Pydantic model configuration"""
        frozen = True  # Make instances immutable
        extra = 'forbid'  # Prevent additional fields

class DecisionSystem1(BaseModel):
    """Base decision model with prediction and confidence"""
    prediction: Prediction
    confidence: int = Field(ge=0, le=100)
    
    class Config:
        frozen = True
        extra = 'forbid'

class DecisionCot(DecisionSystem1):
    """Extended decision model with chain of thought reasoning"""
    risk_factors: List[RiskFactor] = Field(..., min_items=1)



class ResponseProcessor:
    """Handles API response processing and validation"""
    
    def __init__(self, prompt_type: PromptType):
        self.prompt_type = prompt_type
        self.decision_class = DecisionSystem1 if prompt_type == PromptType.SYSTEM1 else DecisionCot
    
    def process_response(self, response_content: str) -> Optional[BaseModel]:
        """
        Processes and validates API response
        
        Args:
            response_content: Raw response string from API
            
        Returns:
            Validated decision model instance
            
        Raises:
            ValidationError: If response fails validation
        """
        try:
            decision = self.decision_class.model_validate_json(response_content)
            self._log_decision(decision)
            return decision
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing response: {e}")
            raise
    
    def _log_decision(self, decision: BaseModel) -> None:
        """Logs processed decision details"""
        logger.info(f"Prediction: {decision.prediction}")
        logger.info(f"Confidence: {decision.confidence}")
        
        if isinstance(decision, DecisionCot):
            for rf in decision.risk_factors:
                logger.info(f"Risk Factor: {rf.factor} ({rf.weight})")



def get_decision(prompt_type: PromptType, model: str = OLLAMA_MODEL) -> Optional[BaseModel]:
    """
    Main function to get and process decision
    
    Args:
        prompt_type: Type of prompt to use
        model: Model identifier for API
        
    Returns:
        Processed decision or None on failure
    """
    processor = ResponseProcessor(prompt_type)
    prompt_str = PROMPT_SYSTEM1 if prompt_type == PromptType.PROMPT_SYSTEM1 else PromptType.PROMPT_COT
    
    try:
        response = chat(
            messages=[{'role': 'user', 'content': prompt_str}],
            model=model,
            format=processor.decision_class.model_json_schema(),
        )
        
        return processor.process_response(response.message.content)
    except Exception as e:
        logger.error(f"Error getting decision: {e}")
        return None

if __name__ == "__main__":
    try:
        decision = get_decision(PromptType.PROMPT_COT)
        if decision:
            print(f"\nPREDICTION: {decision.prediction}")
            print(f"CONFIDENCE: {decision.confidence}")
            
            if isinstance(decision, DecisionCot):
                print("\nRISK FACTORS:")
                for rf in decision.risk_factors:
                    print(f"- {rf.factor} ({rf.weight.value}): {rf.reasoning}")
    except Exception as e:
        logger.error(f"Application error: {e}")



