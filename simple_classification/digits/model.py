import torch.nn as nn

INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 128
OUTPUT_LAYER_SIZE = 10

def build_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=INPUT_LAYER_SIZE, out_features=HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=OUTPUT_LAYER_SIZE)
    )
