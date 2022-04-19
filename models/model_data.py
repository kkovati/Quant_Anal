from dataclasses import dataclass


# a model must contain all the information of the dataset
# e.g. pre, post interval length, investment return threshold


@dataclass
class ModelData:
    pre: int
    post: int
    return_threshold: float
    stop_loss: int


if __name__ == '__main__':
    # Test 1
    md = ModelData(pre=10,
                   post=12,
                   return_threshold=9,
                   stop_loss=90)

    print(md.pre)