import random
import math


class NeuralNetwork:
    def __init__(self, layers):
        # Initialize the neural network with given layers
        self.layers = layers
        self.weights = []  # List to store weights for each layer
        self.biases = []  # List to store biases for each layer

    def randomize(self):
        # Randomly initialize weights and biases for the neural network
        self.weights = []
        self.biases = []
        for layer in range(len(self.layers) - 1):
            # For each layer pair, create weights matrix
            weightsInLayer = []
            for weight in range(self.layers[layer] * self.layers[layer + 1]):
                # Generate a random weight for each connection
                weightsInLayer.append(random.uniform(-1, 1))
            self.weights.append(weightsInLayer)

            # Generate biases only for hidden layers (not the output layer)
            biasesInLayer = []
            if layer != len(self.layers) - 1:
                for bias in range(self.layers[layer + 1]):
                    # Randomly assign biases between -1 and 1
                    biasesInLayer.append(random.uniform(-1, 1))
                self.biases.append(biasesInLayer)

    def mutate(self, rate, change):
        # Mutate weights and biases based on a given mutation rate
        parameters = [self.weights, self.biases]
        for parameter in range(len(parameters)):
            for layer in range(len(parameters[parameter])):
                for value in range(len(parameters[parameter][layer])):
                    if random.random() < rate:
                        # Randomly change the value if the mutation rate allows
                        changeValue = random.uniform(-change, change)
                        parameters[parameter][layer][value] += changeValue
                        if abs(parameters[parameter][layer][value]) > 1:
                            parameters[parameter][layer][value] = (parameters[parameter][layer][value]/abs(parameters[parameter][layer][value]))

    def copy(self):
        copiedNetwork = NeuralNetwork(self.layers)
        copiedNetwork.weights = self.weights
        copiedNetwork.biases = self.biases
        return copiedNetwork

    # Activation functions are provided as static methods for simplicity
    @staticmethod
    def Linear(value):
        return value

    @staticmethod
    def ReLU(value):
        return max(0, value)

    @staticmethod
    def LeakyReLU(value):
        return max(value * 0.05, value)

    @staticmethod
    def Sigmoid(value):
        # Sigmoid function to squash values between 0 and 1
        return 1 / (1 + (2.718 ** (-value)))

    @staticmethod
    def Tanh(value):
        # Hyperbolic tangent function to map values between -1 and 1
        return math.tanh(value)

    def run(self, inputs, activation):
        # current input layer being ran
        currentInput = inputs
        # Loops through all layers but output
        for layer in range(len(self.layers) - 1):
            # resetting calculated output list from last run (instantiating new output list if it is the first run)
            currentOutput = []
            # looping through the current output layer
            for output in range(self.layers[layer + 1]):
                # resetting calculated value for the output node (instantiating new output node list if it is the first run)
                nodeValue = 0
                # looping through the current input layer
                for input in range(self.layers[layer]):
                    # adding up all the node values per each input to the current output node
                    nodeValue += currentInput[input] * self.weights[layer][self.layers[layer + 1] * input + output]
                # appending the calculated sum of output node to the output layer to be used as input in the next run
                currentOutput.append(activation(nodeValue + self.biases[layer][output]))
            # setting the currentOutput to the next run's input
            currentInput = currentOutput
        # once all layers have been iterated through
        return currentInput
