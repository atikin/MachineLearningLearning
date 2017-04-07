import numpy as np
import Perceptron


# TODO: change the fit interface. fit should take a data vector and a label vector
class LinearDiscriminantGauß:
    mean: np.array
    variance: np.array
    num_classes: int
    dimension: int

    def __init__(self, num_classes, feature_dim):
        self.mean = np.empty((num_classes, feature_dim))
        self.variance = np.empty((num_classes, feature_dim, feature_dim))
        self.num_classes = num_classes
        self.dimension = feature_dim

    def fit(self, data):

        for i, label in enumerate(data):
            self.mean[i] = np.mean(label, axis=0)
            self.variance[i] = np.cov(np.transpose(label))

    def predict(self, data):
        labels = np.empty((data.shape[0]))
        for i, instance in enumerate(data):
            classification = np.empty(self.num_classes)
            for clss in range(self.num_classes):
                classification[clss] = self.__discriminator(instance, clss)
            labels[i] = np.argmax(classification)
        return labels

    def __discriminator(self, x, clss):
        return (-0.5 * np.dot((x - self.mean[clss]), np.linalg.inv(self.variance[clss]).dot(x - self.mean[clss]))
                - (self.dimension / 2) * np.log(2 * np.pi)
                - 0.5 * np.log(np.linalg.det(self.variance[clss]))
                + 0)


def mahalanobis_distance(x, mean, variance):
    return np.dot((x - mean), np.linalg.inv(variance).dot(x - mean))
