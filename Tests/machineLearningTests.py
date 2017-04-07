import machineLearning as ml
import numpy as np

def testEuclidDistance():
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 3, 2, 1])
    if ml.euclidDistance(a, b) != np.sqrt(9 + 1 + 1 + 9):
        print("Error in euclid Distance")
    print("Test: Euclid distance succecfull")


def Test():
    testEuclidDistance()

    numOfSamples = 10000
    mean1 = np.array([5, 5])
    covarianz1 = np.array([[5, 3],
                             [3, 5]])
    sampledGauß1= np.random.multivariate_normal(mean1, covarianz1, numOfSamples)

    mean2 = np.array([-5, -5])
    covarianz2 = np.array([[2, 1],
                           [3, 4]])
    sampledGauß2 = np.random.multivariate_normal(mean2, covarianz2, numOfSamples)

    correct = 0
    num = 2 * numOfSamples
    for x in sampledGauß1:
        if ml.linearDiscriminantGauß(x, 0.5, mean1, covarianz1, 2) - ml.linearDiscriminantGauß(x, 0.5, mean2, covarianz2, 2) > 0:
            correct += 1
    for x in sampledGauß2:
        if ml.linearDiscriminantGauß(x, 0.5, mean1, covarianz1, 2) - ml.linearDiscriminantGauß(x, 0.5, mean2, covarianz2, 2) < 0:
            correct += 1

    print(correct / num)
Test()