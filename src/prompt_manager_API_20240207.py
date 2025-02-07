import logging
from typing import Dict, List
import random

from config import Config
from data_manager import DataManager
from models import PromptType

class PromptManager:
    """Manages dynamic prompt generation and templating"""
    def __init__(self, config: Config, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self._base_prompts = {
            PromptType.SYSTEM1: config.prompts.system1,
            PromptType.COT: config.prompts.cot,
            PromptType.COT_NSHOT: config.prompts.cot_nshot
        }
        self._nshot_examples = None  # Cache for n-shot examples
        
    def _generate_nshot_examples(self) -> str:
        """Generate n-shot example string from validation set"""
        if self._nshot_examples is not None:
            return self._nshot_examples
            
        # Get validation data
        validate_data = self.data_manager.get_batch(None, dataset='validate')
        
        # Determine number of examples to use
        n = min(self.config.nshot_ct, len(validate_data))
        
        # Randomly select examples
        selected_examples = random.sample(validate_data, n)
        
        # Build example string
        examples_str = ""
        for i, example in enumerate(selected_examples, 1):
            examples_str += f"""
Example #{i}:
Risk Factors: {example['risk_factors']}
Outcome: {example['target']}
"""
        
        # Cache the examples
        self._nshot_examples = examples_str
        return examples_str
    
    def get_prompt(self, prompt_type: PromptType, row_id: int) -> str:
        """
        Get a prompt with dynamically inserted risk factors from data.
        Now supports n-shot examples for COT_NSHOT prompt type.
        """
        if prompt_type not in self._base_prompts:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
            
        base_prompt = self._base_prompts[prompt_type]
        
        try:
            # Get risk factors text from training data
            risk_factors = self.data_manager.get_risk_factors(row_id)
            
            # Handle n-shot examples if needed
            if prompt_type == PromptType.COT_NSHOT:
                nshot_examples = self._generate_nshot_examples()
                prompt = base_prompt.replace('{nshot_example_str}', nshot_examples)
            else:
                prompt = base_prompt
                
            # Replace risk factors placeholder
            prompt = prompt.replace('{risk_factors}', risk_factors)
            prompt = prompt.replace('###RISK_FACTORS:', risk_factors)
            
            logging.debug(f"Generated {prompt_type} prompt for row {row_id}")
            return prompt
            
        except Exception as e:
            logging.error(f"Error generating prompt for row {row_id}: {str(e)}")
            raise