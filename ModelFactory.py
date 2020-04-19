from ConfigurationType import ConfigurationType 
from Configuration import Configuration
from TensorFlowModelBuilder import TensorFlowModelBuilder
from TensorflowModel import TensorflowModel

class ModelFactory(): 

    def CreateNew(self, configurationType: ConfigurationType, configuration: Configuration):

        if (configurationType == ConfigurationType.tf):
            
            #create the models 
            tfBuilder = TensorFlowModelBuilder() 
            model = tfBuilder.BuildModel(configuration) 
            return TensorflowModel(model) 

        elif (configurationType == ConfigurationType.pytorch):
            pass
            #pytorchBuilder = 
        
       