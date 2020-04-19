from interface import implements
from IModel import IModel

class PytorchModel(implements(IModel)):
    
    def Prepare(self):
        pass 

    def Fit(self, generator): 
        pass 

    def Test(self, generator):
        pass 