import machineLearning as ml
import numpy as np

#Aufgabe 2

omega1 = np.array([[-5.01, -8.12, -3.68],
                   [-5.43, -3.38, -3.54],
                   [ 1.08, -5.52,  1.66],
                   [ 0.86, -3.78, -4.11],
                   [-2.67,  0.63,  7.39],
                   [ 4.94,  3.29,  2.08],
                   [-2.51,  2.09, -2.59],
                   [-2.25, -2.13, -6.94],
                   [ 5.56,  2.86, -2.26],
                   [ 1.03, -3.33, 4.33]])

omega2 = np.array([[-0.91, -0.18, -0.05],
                   [  1.3, -2.06, -3.53],
                   [-7.75, -4.54, -0.95],
                   [-5.47,  0.50,  3.93],
                   [ 6.14,  5.72, -4.85],
                   [ 3.60,  1.26,  4.36],
                   [ 5.37, -4.63, -3.65],
                   [ 7.18,  1.46, -6.66],
                   [-7.39,  1.17,  6.30],
                   [-7.50, -6.32, -0.31]])

# dichotomizer with just the first feature
priorOmega1 = 0.5
priorOmega2 = 0.5
meanOmega1 = np.mean(omega1[0:10:1, 0])
meanOmega2 = np.mean(omega2[0:10:1, 0])
print("mean omega 1: " + str(meanOmega1))
print("mean omega 2: " + str(meanOmega2))

varianceOmega1 = np.var(omega1[0:10:1, 0])
varianceOmega2 = np.var(omega2[0:10:1, 0])

print("variance omega 1: " + str(varianceOmega1))
print("variance omega 2: " + str(varianceOmega2))


def dichotomizer(x):
    return int(ml.linearDiscriminantGauß(x, priorOmega1, meanOmega1, varianceOmega1, 1) -
               ml.linearDiscriminantGauß(x, priorOmega2, meanOmega2, varianceOmega2, 1) > 0)


numOfElements = len(omega2) + len(omega1)
correct = 0
for x in omega1[0:10:1, 0]:
    if dichotomizer(x) == 1:
        correct += 1
for x in omega2[0:10:1, 0]:
    if dichotomizer(x) == 0:
        correct += 1
print("Accuracy: " + str(correct / numOfElements))

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("two features")

#with 2 features

priorOmega1 = 0.5
priorOmega2 = 0.5

meanOmega1 = np.mean(omega1[0:10:1, 0:2:], axis=0)
meanOmega2 = np.mean(omega2[0:10:1, 0:2:], axis=0)
print("mean vector omega 1: " + str(meanOmega1))
print("mean vector omega 2: " + str(meanOmega2))

varianceOmega1 = np.cov(np.transpose(omega1[0:10:1, 0:2:]))
varianceOmega2 = np.cov(np.transpose(omega2[0:10:1, 0:2:]))

print("variance matrix omega 1: " + str(varianceOmega1))
print("variance matrix omega 2: " + str(varianceOmega2))

numOfElements = len(omega2) + len(omega1)
correct = 0
for x in omega1[0:10:1, 0:2]:
    if int(ml.linearDiscriminantGauß(x, priorOmega1, meanOmega1, varianceOmega1, 2) -
           ml.linearDiscriminantGauß(x, priorOmega2, meanOmega2, varianceOmega2, 2) > 0) == 1:
        correct += 1
for x in omega2[0:10:1, 0:2]:
    if int(ml.linearDiscriminantGauß(x, priorOmega1, meanOmega1, varianceOmega1, 2) -
           ml.linearDiscriminantGauß(x, priorOmega2, meanOmega2, varianceOmega2, 2) > 0) == 0:
        correct += 1
print("Accuracy: " + str(correct / numOfElements))

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("three features")
priorOmega1 = 0.5
priorOmega2 = 0.5

meanOmega1 = np.mean(omega1[0:10:1, 0:3:], axis=0)
meanOmega2 = np.mean(omega2[0:10:1, 0:3:], axis=0)
print("mean vector omega 1: " + str(meanOmega1))
print("mean vector omega 2: " + str(meanOmega2))

varianceOmega1 = np.cov(np.transpose(omega1[0:10:1, 0:3:]))
varianceOmega2 = np.cov(np.transpose(omega2[0:10:1, 0:3:]))

print("variance matrix omega 1: " + str(varianceOmega1))
print("variance matrix omega 2: " + str(varianceOmega2))

numOfElements = len(omega2) + len(omega1)
print(numOfElements)
correct = 0
for x in omega1[0:10:1, 0:3]:
    if int(ml.linearDiscriminantGauß(x, priorOmega1, meanOmega1, varianceOmega1, 3) -
           ml.linearDiscriminantGauß(x, priorOmega2, meanOmega2, varianceOmega2, 3) > 0) == 1:
        correct += 1
for x in omega2[0:10:1, 0:3]:
    if int(ml.linearDiscriminantGauß(x, priorOmega1, meanOmega1, varianceOmega1, 3) -
           ml.linearDiscriminantGauß(x, priorOmega2, meanOmega2, varianceOmega2, 3) > 0) == 0:
        correct += 1
print("Accuracy: " + str(correct / numOfElements))