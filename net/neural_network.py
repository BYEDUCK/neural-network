import numpy as np
from random import randint


class NeuralNetwork:
    numberOfInputNeurons = 0
    numberOfHiddenNeurons = 0
    numberOfOutputNeurons = 1
    beta = 0
    eta = 0
    iter = 0
    errors = []

    def __init__(self, nofIN, nofHN, beta, eta, bias=True):
        self.numberOfInputNeurons = nofIN
        self.numberOfHiddenNeurons = nofHN
        self.beta = beta
        self.eta = eta
        self.bias = bias

        # 1
        self.weights_I2H = (np.random.random_sample(
            (self.numberOfHiddenNeurons, self.numberOfInputNeurons))) / 8  # <0 ; 1/8)
        self.weights_H2O = (np.random.random_sample(self.numberOfHiddenNeurons)) / 8

    @staticmethod
    def f(arg, beta):
        return 1 / (1 + np.exp(-beta * arg))

    @staticmethod
    def df(arg, beta):
        y = NeuralNetwork.f(arg, beta)
        return beta * y * (1 - y)

    def countHiddenNeuronsNets(self, inputData):
        hNets = np.empty(self.numberOfHiddenNeurons)
        if self.bias:
            for h in range(self.numberOfHiddenNeurons):
                hNets[h] = np.dot(self.weights_I2H[h, :], inputData) + 1
        else:
            for h in range(self.numberOfHiddenNeurons):
                hNets[h] = np.dot(self.weights_I2H[h, :], inputData)
        return hNets

    def countHiddenNeuronsOuts(self, hNets):
        return NeuralNetwork.f(hNets, self.beta)

    def countOutputNeuronsNets(self, hOuts):
        return np.dot(self.weights_H2O, hOuts)

    def countOutputFromInput(self, inputData):
        hNets = self.countHiddenNeuronsNets(inputData)
        hOuts = self.countHiddenNeuronsOuts(hNets)
        oNets = self.countOutputNeuronsNets(hOuts)
        out = NeuralNetwork.f(oNets, self.beta)
        return out

    def countOutputFromNet(self, oNets):
        return NeuralNetwork.f(oNets, self.beta)

    def test(self, data, desired):
        nofData = data.shape[0]
        output = np.empty(nofData)
        for j in range(nofData):
            output[j] = self.countOutputFromInput(data[j])
        return (np.sum(np.power((output - desired), 2))) / nofData, output

    def train(self, inputData, target, maxIter, tolerance):
        nofSets = inputData.shape[0]
        for it in range(maxIter):
            err, _ = self.test(inputData, target)
            if err <= tolerance:
                break
            self.errors.append(err)
            self.iter += 1

            # 2
            drawnInputIndex = randint(0, nofSets - 1)
            drawnInput = inputData[drawnInputIndex, :]

            # 3
            hNets = self.countHiddenNeuronsNets(drawnInput)

            # 4
            hOuts = self.countHiddenNeuronsOuts(hNets)

            # 5
            oNets = self.countOutputNeuronsNets(hOuts)

            # 6 *stan wyjsc warstwy wyjsciowej
            out = self.countOutputFromNet(oNets)

            desiredOutput = target[drawnInputIndex]

            # 11 test
            # err, output = net.test(inputData, target)
            # if err < tolerance:
            # break

            # 7
            outputLayerErrorSignal = (desiredOutput - out) * NeuralNetwork.df(oNets, self.beta)

            hiddenLayerErrorSignal = np.empty(self.numberOfHiddenNeurons)

            # 8, 9
            for h in range(self.numberOfHiddenNeurons):
                hiddenLayerErrorSignal[h] = NeuralNetwork.df(hNets[h], self.beta) \
                                            * self.weights_H2O[h] * outputLayerErrorSignal
                self.weights_H2O[h] += self.eta * outputLayerErrorSignal * hOuts[h]
            # 10
            for h in range(self.numberOfHiddenNeurons):
                for i in range(self.numberOfInputNeurons):
                    self.weights_I2H[h, i] += self.eta * hiddenLayerErrorSignal[h] * drawnInput[i]
