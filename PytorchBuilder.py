from interface import implements
from IModelBuilder import IModelBuilder 
from Configuration import Configuration
from LayerTypes import LayerTypes 

class TensorFlowModelBuilder(implements(IModelBuilder)): 

    def GetModelLayerStep(self, layer):
        pass

    def BuildModel(self, configuration : Configuration):
        pass