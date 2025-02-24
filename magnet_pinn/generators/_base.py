from abc import ABC

class BaseGenerator(ABC):
    def __init__(self, **kwargs):
        pass

    def generate(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.generate(**kwargs)