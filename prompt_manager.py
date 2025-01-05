# prompt_manager.py
import logging
from typing import Dict

from config_ver6 import Config
from data_manager import DataManager
from models import PromptType

class PromptManager:
    """Manages dynamic prompt generation and templating"""
    def __init__(self, config: Config, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self._base_prompts = {
            PromptType.SYSTEM1: config.prompts["system1"],
            PromptType.COT: config.prompts["cot"]
        }
    
    def get_prompt(self, prompt_type: PromptType, row_id: int) -> str:
        """
        Get a prompt with dynamically inserted risk factors from data.
        
        Args:
            prompt_type: Type of prompt to generate
            row_id: ID of the data row to use for risk factors
            
        Returns:
            Formatted prompt string with risk factors inserted
        
        Raises:
            ValueError: If the prompt type is invalid or row_id doesn't exist
            RuntimeError: If the data hasn't been loaded
        """
        if prompt_type not in self._base_prompts:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
            
        base_prompt = self._base_prompts[prompt_type]
        
        try:
            # Get risk factors text from training data
            risk_factors = self.data_manager.get_risk_factors(row_id)
            
            # Replace placeholder with actual risk factors
            # We use both formats to ensure compatibility
            prompt = base_prompt.replace('{risk_factors}', risk_factors)
            prompt = prompt.replace('###RISK_FACTORS:', risk_factors)
            
            logging.debug(f"Generated {prompt_type} prompt for row {row_id}")
            return prompt
            
        except Exception as e:
            logging.error(f"Error generating prompt for row {row_id}: {str(e)}")
            raise