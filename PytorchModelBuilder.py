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
                self.Layers.append(nn.MaxPool2d(layer.KernelSize, layer.Stride))
            else:
                self.Layers.append(None) #use this as a placeholder. We likely defined a layer than wont be implemented until forward() prop 

    def forward(self, x):
        
        for layer_index, layer in enumerate(self.Configuration.Layers):      

            currentLayer = self.Layers[layer_index]

            if (layer.LayerType == LayerTypes.Activation):
                if (layer.ActivationType == 'relu'):
                    x = F.relu(x)
                elif (layer.ActivationType == 'sigmoid'):
                    x = F.sigmoid(x)

            elif (layer.LayerType == LayerTypes.Flatten):
                x = x.view(-1, layer.DimensionToReduce * self.Configuration.BatchSize)
            
            else:
                x = currentLayer(x) #otherwise append the layer as is.

        return x
