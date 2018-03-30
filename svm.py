###################
# SVM classifier #
###################

import csv
import math
import operator
import numpy


### LOAD DATASET
def loadDataset(filename):
    with open(filename, 'r') as csvfile:
        rows = list(csv.reader(csvfile))
        return numpy.array([[int(i) for i in row] for row in rows])


### SIMILARITY MEASURE
def euclideanDistance(instance1, instance2):
    return numpy.linalg.norm(instance1 - instance2)


def hammingDistance(instance1, instance2):
    return numpy.count_nonzero(instance1 != instance2)


### DETERMINE PREDICTION ACCURACY
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


### MAIN
def main():
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    # generate predictions
    predictions = []
    for x in range(len(testSet)):
        predictions = 1

    # compute prediction accuracy
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


# get data
trainingSet = loadDataset('train.csv')
testSet = loadDataset('test.csv')
testSet = testSet[:100]

main()