from ConfigurationType import ConfigurationType 
from Configuration import Configuration
from TensorFlowModelBuilder import TensorFlowModelBuilder

class ModelFactory(): 

    def CreateNew(self, configurationType: ConfigurationType, configuration: Configuration):

        if (configurationType == ConfigurationType.tf):
            
            #create the models 
            tfBuilder = TensorFlowModelBuilder() 
            return tfBuilder.BuildModel(configuration) 
        
       