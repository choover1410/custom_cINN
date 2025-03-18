import torch.nn as nn

def convolution_network(Hidden_layer):
    return lambda input_channel, output_channel: nn.Sequential(
                                    nn.Conv2d(input_channel, Hidden_layer, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(Hidden_layer, output_channel, 3, padding=1))

def fully_connected(Hidden_layer):
    return lambda input_data, output_data: nn.Sequential(
                                    nn.Linear(input_data, Hidden_layer),
                                    nn.ReLU(),
                                    nn.Linear(Hidden_layer, output_data))