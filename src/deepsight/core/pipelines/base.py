from abc import ABC, abstractmethod

class Step(ABC):
    
    @abstractmethod
    def run(self,*args,context:dict,**kwargs)->dict:
        pass

class Pipeline:
    def __init__(self,steps:list[Step]):
        self.steps = steps
        self.context = {}
    def run(self,**kwargs)->dict:
        self.context.update(kwargs)
        for step in self.steps:
            step.run(context=self.context)
        return self.context