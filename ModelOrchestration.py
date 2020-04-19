from ConfigurationReader import ConfigurationReader
from ModelFactory import ModelFactory 
from ConfigurationType import ConfigurationType 
from ImageDataGenerator import DataGenerator
from DataReader import DataReader

#define paths / constants
ConfigurationPath = 'Configuration.json'

#read in the model configuration from the json 
ModelConfiguration = ConfigurationReader().ReadConfiguration(ConfigurationPath) 


tf_model = ModelFactory().CreateNew(ConfigurationType.tf, ModelConfiguration)


#setup data generation 
reader = DataReader()
(celeba_partition, celeba_labels) = reader.read_celeb_a()
training_gen = DataGenerator(celeba_partition['train'] + celeba_partition['validation'], celeba_labels, ModelConfiguration)
test_gen = DataGenerator(celeba_partition['test'], celeba_labels, ModelConfiguration)

tf_model.Prepare()
tf_model.Fit(training_gen)
tf_model.Test(test_gen) 
