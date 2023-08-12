import matplotlib.pyplot as plt
from copy import deepcopy

def plot_trellis_with_path(trellis, path) -> None:
    """
    helper function to display path on trellis
    """

    # To plot trellis with path, taking advantage of 'nan' value
    trellis_with_path = deepcopy(trellis)
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.show()
