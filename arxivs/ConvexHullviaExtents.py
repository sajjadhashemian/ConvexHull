import numpy as np
from common import normalize

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class ConvexHullviaExtents:
    """
    Class to compute and store the convex hull extents.
    """

    def __init__(self, points):
        self.points = np.array(points)
        self.X = np.array(points)
        self.extents, self.random_extents, self.extents_max = None, None, None

    def get_extents(self, t=1000, return_max_extent=False):
        """
        Calculates the directions of the convex hull's extents in 2D.
        """
        directions = np.array(
            [
                np.array([np.cos(theta), np.sin(theta)])
                for theta in np.linspace(0, 2 * np.pi, t)
            ]
        )
        extents = dict([[tuple(x), []] for x in self.X])

        dot_products = np.dot(self.X, directions.T)
        for i, direction in enumerate(directions):
            x = self.X[np.argmax(dot_products[:, i])]
            extents[tuple(x)].append(direction)
        if return_max_extent:
            extents_max = dict([[tuple(x), []] for x in self.X])
            for i, x in enumerate(self.X):
                extents_max[tuple(x)] = normalize(
                    directions[np.argmax(dot_products[i, :])]
                )

            return extents, extents_max
        return extents

    def get_random_extents(self, t, return_approx_hull=False):
        """
        Calculates the randomized directional extent of the convex hull's extents in 2D.
        """
        random_directions = np.random.normal(size=(t, 2))
        random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
        random_extents = dict({})
        for x in self.X:
            random_extents[tuple(x)] = []
        random_dot_products = np.dot(self.X, random_directions.T)
        for i, random_directions in enumerate(random_directions):
            x = self.X[np.argmax(random_dot_products[:, i])]
            random_extents[tuple(x)].append(random_directions)

        if return_approx_hull:
            random_extents_set = set()
            for x in self.X:
                if len(random_extents[tuple(x)]) > 0:
                    random_extents_set.add(tuple(x))
            return random_extents, random_extents_set
        else:
            return random_extents
