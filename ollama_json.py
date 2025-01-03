import requests
import json

from ollama import chat
from pydantic import BaseModel

OLLAMA_MODEL = 'Phi-4:Q4_K_M' # 'Phi-4:Q4_K_M' or 'llama3.2:3b', 

class decision_system1(BaseModel):
  prediction: list[str]
  confidence: int

class decision_cot(BaseModel):
  prediction: list[str]
  confidence: int
  reasoning: list[str]

class decision_cot_nshot(BaseModel):
  prediction: list[str]
  confidence: int

DECISION = decision_system1

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
3. a list of text strings identifying each risk factor with reasoning to weight each risk factor

###INSTRUCTIONS:
Please response with only a one word “YES” or “NO”, an integer 0-100 confidence percentage, and a list of text strings with reasoning weighting each risk factor for your 3 year recidivism prediction. DO NOT say 'Hello' or any greetings. Use your judicial and statistical expertise with these ###RISK_FACTORS to predict if the person will be rearrested within the next 3 years. Do not provide any description or reasoning. ONLY respond with “YES” or “NO” for your 3 year recidivism prediction. Return your responses in the form of ###RESPONSE_FORMAT. DO NOT say 'Hello' or any other greetings.
"""

PROMPT_STR = prompt_cot # prompt_system1

response = requests.post('http://localhost:11434/api/generate', json={
  "model": OLLAMA_MODEL,
  "prompt": PROMPT_STR,
  "stream": False,
  "format": "json"
})

print(json.loads(response.content)['response'])