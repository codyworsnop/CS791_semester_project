from interface import Interface
from Configuration import Configuration

class IModelBuilder(Interface): 

    def BuildModel(self, configuration : Configuration):
        pass 

    