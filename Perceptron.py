import matplotlib.pyplot as plt
import numpy as np
import plot


#TODO extend to multiclass
#TODO implement more training algorithms
class Perceptron2Class:

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


def test():
    num_of_samples = 100
    mean1 = np.array([5, 5])
    mean2 = np.array([-5, -5])
    covarianz1 = np.array([[3, 2],
                           [2, 4]])
    covarianz2 = np.array([[3, 2],
                           [2, 2]])
    sampled_gauß_1 = np.random.multivariate_normal(mean1, covarianz1, num_of_samples)
    sampled_gauß_2 = np.random.multivariate_normal(mean2, covarianz2, num_of_samples)

    labels = np.array([x for x in range(num_of_samples * 2)])

    for i, x in enumerate(labels):
        if i >= num_of_samples:
            labels[i] = 2
        else:
            labels[i] = 1

    classifier = Perceptron2Class(0.001)
    classifier.fit(np.concatenate((sampled_gauß_1, sampled_gauß_2)), labels )

    plot.plot_decision_regions(np.concatenate((sampled_gauß_1, sampled_gauß_2)), labels, classifier)
    plt.show()

test()
