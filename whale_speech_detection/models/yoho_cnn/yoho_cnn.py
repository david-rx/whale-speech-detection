
from torch.nn import Linear, Module, Conv1d, Conv2d, Sequential, CrossEntropyLoss, MaxPool2d, Softmax, BatchNorm2d, ReLU
import torch

LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    ([3, 3], 1,   64),
    ([3, 3], 2,  128),
    ([3, 3], 1,  128),
    ([3, 3], 2,  256),
    ([3, 3], 1,  256),
    ([3, 3], 2,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 2, 1024),
    ([3, 3], 1, 1024),
    ([3, 3], 1, 512),
    ([3, 3], 1, 256),
    ([3, 3], 1, 128),
]

class YohoCnn(Module):

    """
    TODO: Work in Progress!! This is copied from
    the YOHO paper and is unlikely to work as is!!
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        for i in range(len(LAYER_DEFS)):
            layer = Sequential(
                Conv2d(1, 1, groups=1, kernel_size=LAYER_DEFS[i][0], stride=LAYER_DEFS[i][1], padding='?'), #make this depthwise
                BatchNorm2d(4),
                ReLU(inplace=True),
                Conv2d(1, 4, kernel_size=LAYER_DEFS[i][0], stride=LAYER_DEFS[i][1], padding='?'), #how many input and output channels?
                BatchNorm2d(4),
                ReLU(inplace=True),
            )
            layers.append(layer)
        return layers

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


