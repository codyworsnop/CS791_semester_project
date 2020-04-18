from interface import implements
from IModelBuilder import IModelBuilder 
from Configuration import Configuration
from LayerTypes import LayerTypes 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

class TensorFlowModelBuilder(implements(IModelBuilder)): 

    def GetModelLayerStep(self, layer):
        layerType = layer.LayerType 

        if (layerType == LayerTypes.Conv2D): 
            return layers.Conv2D(layer.FilterSize, layer.KernelSize, padding=layer.Padding, strides=layer.Strides)
        elif (layerType == LayerTypes.Dense):
            return layers.Dense(layer.Connections)
        elif (layerType == LayerTypes.GlobalAveragePooling2D): 
            return layers.GlobalAveragePooling2D()
        elif (layerType == LayerTypes.Activation):
            return layers.Activation(layer.ActivationType)

    def BuildModel(self, configuration : Configuration):

        input_image = layers.Input(shape=configuration.Shape)
        previousLayer = input_image 

        for layer in configuration.Layers:
            model = self.GetModelLayerStep(layer)(previousLayer)
            previousLayer = model 

        model = models.Model(input_image, previousLayer)

        return model 