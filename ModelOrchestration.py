from ConfigurationReader import ConfigurationReader
from ModelFactory import ModelFactory 
from ConfigurationType import ConfigurationType 
from ImageDataGenerator import DataGenerator
from DataReader import DataReader
import cv2 
import numpy as np
from Visualizer import Visualizer 

#define paths / constants
ConfigurationPath = 'Configuration.json'

#read in the model configuration from the json 
ModelConfiguration = ConfigurationReader().ReadConfiguration(ConfigurationPath) 

factory = ModelFactory()
#tf_model = factory.CreateNew(ConfigurationType.tf, ModelConfiguration)
pyt_model = factory.CreateNew(ConfigurationType.pytorch, ModelConfiguration)

#setup data generation 
reader = DataReader()

(celeba_partition, celeba_labels) = reader.read_celeb_a()
training_gen = DataGenerator((celeba_partition['train'] + celeba_partition['validation']), celeba_labels, ModelConfiguration)
test_gen = DataGenerator(celeba_partition['test'], celeba_labels, ModelConfiguration)
pyt_model.Prepare()
pyt_model.Fit(training_gen)


tf_model.Prepare()
performance_results = tf_model.Fit(training_gen)
accuracy_results = tf_model.Test(test_gen) 

np.save("perf_tf.npy", performance_results)
np.save("perf_tf_accu.npy", accuracy_results)
#tf_data = np.load("perf_tf.npy", allow_pickle=True)
visualizer = Visualizer() 
visualizer.Graph(np.save("perf_tf.npy", performance_results)) 



