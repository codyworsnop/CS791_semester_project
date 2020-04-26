from enum import Enum

class LayerTypes(Enum):
    Conv2D = 0 
    Activation = 1
    GlobalAveragePooling2D = 2
    Dense = 3
    Flatten = 4