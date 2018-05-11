import numpy as np
import os.path
from net import *
import matplotlib.pyplot as plt

def getDataFromCSVFile(path):
    with open(path, 'rt') as csv_file:
        data = np.genfromtxt(csv_file, delimiter=',')
    return data


if __name__ == "__main__":

    # Wczytanie danych
    projectPath = os.path.abspath(os.path.dirname(__file__))
    dataPath = os.path.join(projectPath, "../data.csv")
    myData = getDataFromCSVFile(dataPath)
    numberOfDataSets = myData.shape[0]
    numberOfColumns = myData.shape[1]
    # bias = np.ones((numberOfDataSets, 1)) * 5
    desiredOut = myData[:, -1]

    # Odciecie ostatniej kolumny zawierajacej wyjscia
    inputs = np.delete(myData, np.s_[numberOfColumns - 1], axis=1)

    # inputs = np.append(inputs, bias, axis=1)
    print("Inputs matrix size: " + repr(inputs.shape))
    print("Target matrix size: " + repr(desiredOut.shape))

    # Wybranie i potasowanie indeksow wektorow wejsciowych do uczenia i testowania
    testingIndexes = [i for i in range(0, numberOfDataSets - 1, 2)]
    teachingIndexes = [i for i in range(1, numberOfDataSets, 2)]
    np.random.shuffle(teachingIndexes)
    np.random.shuffle(testingIndexes)
    teachingData = inputs[teachingIndexes, :]
    testingData = inputs[testingIndexes, :]
    desiredTestOut = desiredOut[testingIndexes]
    desiredTeachOut = desiredOut[teachingIndexes]

    # Utworzenie sieci i uruchomienie algorytmu uczenia
    net = NeuralNetwork(teachingData.shape[1], 60, 1, 0.3)

    print("\ninit I2H weights:\n")
    print(net.weights_I2H)
    print("\ninit H2O weights:\n")
    print(net.weights_H2O)
    net.train(teachingData, desiredTeachOut, 20000, 0.08)

    iters = range(1, net.iter + 1)
    plt.figure(1)
    plt.plot(iters, net.errors, 'r-')
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    print("\nI2H weights:\n")
    print(net.weights_I2H)
    print("\nH2O weights:\n")
    print(net.weights_H2O)

    # Test sieci
    finalErr, finalOutput = net.test(testingData, desiredTestOut)

    print("\nAlgorithm ended in %d iterations\n" % net.iter)
    print("\nfinal error: %.5f\n" % finalErr)
    print("\noutput:\n")
    print(finalOutput)
    print("\ndesired output:\n")
    print(desiredTestOut)
    print("\nroznica:\n")
    print(desiredOut[testingIndexes] - finalOutput)
