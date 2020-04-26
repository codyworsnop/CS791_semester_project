from interface import implements
from IModelBuilder import IModelBuilder 
from Configuration import Configuration
from LayerTypes import LayerTypes 

import torch.nn as nn
import torch.nn.functional as F
import torch 

class PytorchModelBuilder(implements(IModelBuilder), nn.Module): 

    def __init__(self, configuration):
        super(PytorchModelBuilder, self).__init__()
        self.Configuration = configuration

        self.Layers = nn.ModuleList()

    def BuildModel(self, configuration : Configuration):
        
        for layer in configuration.Layers:
            
            layerType = layer.LayerType 

            if (layerType == LayerTypes.Conv2D): 
                self.Layers.append(nn.Conv2d(layer.ChannelIn, layer.ChannelOut, layer.KernelSize, layer.Strides))
            elif (layerType == LayerTypes.Dense):
                self.Layers.append(nn.Linear(layer.InputConnections, layer.Connections))
            elif (layerType == LayerTypes.GlobalAveragePooling2D): 
                self.Layers.append(nn.AvgPool2d(layer.KernelSize))

    def forward(self, x):
        
        previousLayer = None
    
        for layer_index, layer in enumerate(self.Configuration.Layers):      

            if (layer.LayerType == LayerTypes.Activation):
                if (self.Configuration.Layers[layer_index - 1].LayerType == LayerTypes.GlobalAveragePooling2D):
                    x = x.view(-1, 40 * 64 * 64)

                if (layer.ActivationType == 'relu'):
                    x = F.relu(previousLayer(x))
                elif (layer.ActivationType == 'sigmoid'):
                    x = F.sigmoid(previousLayer(x))

            previousLayer = self.Layers[layer_index]

        return x
