import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

def conv3x3(in_channels, out_channels, ksize=1, stride=None, **kwargs):
    if stride is None:
        return MetaSequential(
            MetaConv2d(in_channels, out_channels,
                       kernel_size=ksize,
                       padding=1,
                       **kwargs),
            MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    else:
        return MetaSequential(
            MetaConv2d(in_channels, out_channels,
                       stride=[stride,stride],
                       kernel_size=ksize,
                       padding=0,
                       **kwargs),
            MetaBatchNorm2d(out_channels, momentum=1.,
                            track_running_stats=False),
            nn.ReLU(),
        )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features,
                       hidden_size=64, fc_in_size=None,
                       conv_kernel=[3,3,3,3], strides=None):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        if strides is None:
            strides = [None] * len(conv_kernel)
        else:
            assert(len(strides)==len(conv_kernel))

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size, conv_kernel[0], strides[0]),
        )
        for k in range(1, len(conv_kernel)):
            self.features.add_module('block_'+str(k),
                                     conv3x3(hidden_size, hidden_size, conv_kernel[k], strides[k]))

        if fc_in_size is None:
            fc_in_size = hidden_size
        self.classifier = MetaLinear(fc_in_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits
