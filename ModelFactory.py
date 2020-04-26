from ConfigurationType import ConfigurationType 
from Configuration import Configuration
from TensorFlowModelBuilder import TensorFlowModelBuilder
from PytorchModelBuilder import PytorchModelBuilder
from TensorflowModel import TensorflowModel
from PytorchModel import PytorchModel
import torch.optim as optim

class ModelFactory(): 

    def CreateNew(self, configurationType: ConfigurationType, configuration: Configuration):

        if (configurationType == ConfigurationType.tf):
            
            tfBuilder = TensorFlowModelBuilder() 
            model = tfBuilder.BuildModel(configuration) 
            model.summary()
            return TensorflowModel(model) 

        elif (configurationType == ConfigurationType.pytorch):
            
            pytorchBuilder = PytorchModelBuilder(configuration)
            pytorchBuilder.BuildModel(configuration)

            return PytorchModel(pytorchBuilder)
        
       