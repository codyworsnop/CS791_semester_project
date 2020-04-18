from ConfigurationReader import ConfigurationReader
from ModelFactory import ModelFactory 
from ConfigurationType import ConfigurationType 

#define paths / constants
ConfigurationPath = 'Configuration.json'

#read in the model configuration from the json 
ModelConfiguration = ConfigurationReader().ReadConfiguration(ConfigurationPath) 

tf_model = ModelFactory().CreateNew(ConfigurationType.tf, ModelConfiguration)

a = 3 

