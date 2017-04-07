import numpy as np


# just linear seperable problem
class Perceptron2D:

    learning_rate: float
    weights: np.array

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def fit(self, data, labels):
        if data.shape[0] != labels.shape[0]:
            raise Exception(
                "Num of label must equal number of data: {:d} != {:d}".format(data.shape[0], labels.shape[0]))

        self.weights = np.zeros(data.shape[1] + 1)
        transformed_data = self.__transform(data, labels)  # transform for batch perceptron algorithm

        false_classified_data = np.array([x for x in transformed_data if self.__discriminate(x) <= 0])

        while false_classified_data.shape[0] > 0:
            false_classified_data = np.array([x for x in transformed_data if self.__discriminate(x) <= 0])
            print("False classifications {:d}".format(false_classified_data.shape[0]))
            self.weights = self.weights + self.learning_rate * np.sum(false_classified_data, axis=0)

    def __transform(self, data, labels):
        transformed = np.array([np.append([1], x) for x in data])
        for i, x in enumerate(transformed):
            if labels[i] == 2:
                transformed[i] = -x
            else:
                transformed[i] = x
        return transformed

    def predict(self, data):
        labels = np.empty((data.shape[0]))
        transformed_data = np.array([np.append([1], x) for x in data])  # transform for batch perceptron algorithm
        for i, instance in enumerate(transformed_data):
            classification = self.__discriminate(instance)
            if classification > 0:
                labels[i] = 1
            elif classification < 0:
                labels[i] = 2
            else:  # set to 0 if cant decide
                labels[i] = 0
        return labels

    def __discriminate(self, x):
        return np.dot(self.weights, x)


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
