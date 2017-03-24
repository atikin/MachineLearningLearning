import numpy as np


def linearDiscriminantGauß(x, prior, mean, variance, dimension):
    if dimension != 1:
        return (-0.5*np.dot((x - mean), np.linalg.inv(variance).dot(x - mean))
                - (dimension / 2) * np.log(2 * np.pi)
                - 0.5 * np.log(np.linalg.det(variance))
                + np.log(prior))
    else:
        return ((-0.5 * np.power((x - mean), 2) / variance)
                - 0.5 * np.log(2 * np.pi * variance)
                + np.log(prior))


def euclidDistance(x, y):
    if len(x.shape) != 1 and len(y.shape) != 1:
        raise ValueError("x or y is not a vector")
    else:
        return np.sqrt(sum(np.power(x-y, 2)))


def mahalanobisDistance(x, mean, variance):
    return np.dot((x-mean), np.linalg.inv(variance).dot(x - mean))
