import numpy as np
import matplotlib.pyplot as plt
from ConvexHullviaExtents import ConvexHullviaExtents
from ConvexHullviaMVEE import ConvexHullviaMVEE
from common import MVEE

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class plots:
    """
    Class to handle plotting of convex hulls and extents.
    """

    def __init__(self, points, hull):
        self.points = np.array(points)
        self.hull = hull
        # Plot the points
        plt.plot(self.points[:, 0], self.points[:, 1], "o", label="Points")
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        self.plt = plt

    def convex_hull(self):
        """
        Plots the convex hull of a set of points in 2D.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        points, hull, plt = self.points, self.hull, self.plt

        # Convert points to a numpy array
        points = np.array(points)

        # Plot the convex hull simplices
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], "k-")
        plt.scatter(0, 0, c="black", zorder=11111, label="Origin")

        # plt.plot(
        #     points[hull.vertices[:], 0],
        #     points[hull.vertices[:], 1],
        #     "o",
        #     c="r",
        #     label="Hull Vertices",
        # )
        plt.rcParams["figure.figsize"] = (10, 6)

        plt.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            alpha=0.3,
            label="Convex Hull",
        )
        return plt

    def plot_directional_extents(self, t=1000):
        """
        Plots the convex hull and its extents.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        plt = self.plt
        points, plt = self.points, self.plt

        ConvHull = ConvexHullviaExtents(points)
        extents = ConvHull.get_extents(t)

        for point, directions in extents.items():
            for direction in directions:
                plt.arrow(
                    point[0],
                    point[1],
                    direction[0],
                    direction[1],
                    head_width=0,
                    head_length=0,
                    fc="b",
                    ec="b",
                    alpha=0.02,
                    # label="Extents Cones",
                )
        return plt

    def plot_convex_hull_extents(self, t=50):
        """
        Plots the convex hull and its extents.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        plt = self.plt
        points, plt = self.points, self.plt

        ConvHull = ConvexHullviaExtents(points)
        random_extents, random_extents_set = ConvHull.get_random_extents(t, True)
        for point, directions in random_extents.items():
            for direction in directions:
                plt.arrow(
                    point[0],
                    point[1],
                    direction[0],
                    direction[1],
                    head_width=0.1,
                    head_length=0.2,
                    fc="r",
                    ec="r",
                    alpha=0.2,
                    # label="Directional Extents",
                )
        return plt

    def AllviaExtents(self):
        plt = self.plt
        plt = self.convex_hull()
        plt = self.plot_directional_extents()
        plt = self.plot_convex_hull_extents()
        # plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.show()
        return plt

    def plot_MVEE(self):
        """
        Plots the Minimum Volume Enclosing Ellipsoid (MVEE) in 2D.

        Parameters:
        c (array-like): Center of the MVEE.
        A (array-like): Shape matrix of the MVEE.
        """
        plt = self.plt
        cs, Ps = MVEE(self.points)
        step = 0.01
        t = np.arange(0, 2 * np.pi + step, step)
        y = np.vstack((np.cos(t), np.sin(t))).T

        ellip = np.linalg.inv(Ps) @ (y + cs).T
        plt.plot(ellip[0], ellip[1], color="b", label="MVEE Boundary")
        plt.scatter(cs[0], cs[1], c="g", marker="x", label="Center of MVEE")
        return plt

    def plot_MVEE_convex_hull_extents(self, t=2):
        """
        Plots the convex hull and its extents.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        points, plt = self.points, self.plt

        ConvHull = ConvexHullviaMVEE(points)
        S, extents, U = ConvHull.compute(m=t, return_extents=True)

        for u in U:
            plt.arrow(
                0,
                0,
                u[0],
                u[1],
                head_width=0.1,
                head_length=0.2,
                fc="grey",
                ec="grey",
                alpha=0.9,
                zorder=1111111,
                # label="Presampled vectors",
            )

        for point, directions in extents.items():
            for direction in directions:
                plt.arrow(
                    point[0],
                    point[1],
                    direction[0],
                    direction[1],
                    head_width=0.1,
                    head_length=0.2,
                    fc="r",
                    ec="r",
                    alpha=0.2,
                    # label="Directional Extents",
                )
        return plt

    def AllviaMVEE(self):
        plt = self.plt
        plt = self.convex_hull()
        plt = self.plot_directional_extents()
        plt = self.plot_convex_hull_extents()
        plt = self.plot_MVEE()
        plt = self.plot_MVEE_convex_hull_extents()
        # plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.show()
        return plt

    def Plain(self):
        plt = self.plt
        plt = self.convex_hull()
        plt = self.plot_directional_extents()
        # plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.show()
        return plt
