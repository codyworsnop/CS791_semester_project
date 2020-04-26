from typing import List
from LayerTypes import LayerTypes

class Layer():

    def __init__(self, layerType, obj):
        self.LayerType = LayerTypes(layerType)

        if (self.LayerType == LayerTypes.Conv2D):
            self.FilterSize = obj['FilterSize']
            self.KernelSize = (obj['KernelSize_x'], obj['KernelSize_y']) 
            self.Padding = obj['Padding']
            self.Strides = (obj['Strides_x'], obj['Strides_y'])
            self.ChannelIn = obj['Channel_in']
            self.ChannelOut = obj['Channel_out']
        elif (self.LayerType == LayerTypes.Activation):
            self.ActivationType = obj['ActivationType']
        elif (self.LayerType == LayerTypes.Dense):
            self.Connections = obj['Connections']
            self.InputConnections = obj['InputConnections']
        elif (self.LayerType == LayerTypes.GlobalAveragePooling2D):
            self.KernelSize = obj['Kernel_size']

class Configuration(): 

    def __init__(self, shape_x, shape_y, shape_z, layers: List[Layer]):
        self.Shape = (shape_x, shape_y, shape_z)
        self.Layers = layers
