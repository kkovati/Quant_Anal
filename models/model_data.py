from dataclasses import dataclass


# a model must contain all the information of the dataset
# e.g. pre, post interval length, investment return threshold


@dataclass
class ModelData:
    pre: int
    post: int
    return_threshold: float
    stop_loss: int

