import numpy as np
from typing import Tuple
from .classification import label_to_discrete


__all__ = ["label_to_survival", "survival_to_label"]


def label_to_survival( label, classes: tuple, max_time: int) -> np.ndarray:
    event = label[0]  # label_to_discrete( label[0], classes )  #

    floor_time = np.floor(label[1])
    time = min(floor_time, max_time-1)

    # if alive after max time -> censored
    if event != classes[0] and time < floor_time:
        event = classes[0]

    return np.array([ label_to_discrete(event, classes), time ], dtype=int)


def survival_to_label( label, classes: tuple, max_time: int, **_ ) -> Tuple[str, int]:
    # assert len(label) == len( classes ) * max_time

    time = np.argmax( label )
    event = 0
    while time > max_time:
        time -= max_time
        event += 1
    return classes[ event ], time


if __name__ == "__main__":
    pass
