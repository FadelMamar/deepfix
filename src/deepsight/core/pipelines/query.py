from typing import List,Union,Optional,Dict,Any
from .base import Step
from ..query import PromptBuilder

class PromptBuilder(Step):
    def __init__(self):
        self.builder = PromptBuilder()
    
    def run(
        self,
        context:dict,
        artifacts:Optional[List]=None,
        query_context:Optional[Dict[str, Any]]=None,
    ) -> dict:        
        prompt = self.builder.build_prompt(artifacts=artifacts or context.get("artifacts"),context=query_context)
        context['prompt'] = prompt
        return context