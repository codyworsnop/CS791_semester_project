import json
from Configuration import Layer
from Configuration import Configuration

class ConfigurationReader(): 

    def object_decoder(self, obj): 

        if 'LayerType' in obj: 
            return Layer(obj['LayerType'], obj)
        else:
            return Configuration(obj['Shape_x'], obj['Shape_y'], obj['Shape_z'], obj['Batch_size'], obj['Layers'])

    def ReadConfiguration(self, configurationPath : str):
  
        #read the json
        with open(configurationPath, 'r') as read_file:
            data = json.load(read_file, object_hook=self.object_decoder)

        return data 