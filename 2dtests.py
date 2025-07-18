import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from common import sample_input
from plots import plots

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


if __name__ == "__main__":
    # m = 20
    # kappa = 10.0  # higher → more tightly around e1
    # S = convex_hull_via_mvee(_Z, m, kappa)

    _Z = np.random.normal(loc=(0, 0), scale=0.5, size=(40, 2))
    hull = ConvexHull(_Z)
    plt = plots(_Z, hull)
    x = plt.Plain()
    # x = plt.AllviaExtents()
    # x = plt.AllviaMVEE()
    x.show()
