from typing import Literal

GPU = Literal[0, 1, 2, 3, 4, 5, 6, 7]
ONLY_TRAIN = Literal["baselines", "miners"]
MODEL_TYPE = Literal[
    "single-neuron-perceptron",
    "two-layer-perceptron",
    "simple-cnn",
    "resnet18",
    "densenet",
    "deit-b",
]
NORM_TYPE = Literal["batch", "group"] | None
