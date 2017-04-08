import numpy as np
import matplotlib.pyplot as plt


class CostFunction:

    function: object

    function_diff: object

    def __init__(self, fnc, fnc_diff):
        self.function = fnc
        self.function_diff = fnc_diff

    def eval(self, x, y):
        return self.function(x, y)

    def eval_dif(self, x, y):
        return self.function_diff(x, y)


class ActivationFunction:
    function: object

    function_diff: object

    def __init__(self, fnc, fnc_diff):
        self.function = fnc
        self.function_diff = fnc_diff

    def eval(self, x):
        return self.function(x)

    def eval_diff(self, x):
        return self.function_diff(x)

class Layer:

    weights: np.array

    output_before_ac: np.array
    input: np.array

    activation: ActivationFunction

    #TODO initialize the bias independent from the weights
    def __init__(self, num_input, num_hidden_units, activation):
        self.weights = np.random.normal(0, 0.3, (num_hidden_units, num_input + 1))
        self.output_before_ac = np.zeros(num_hidden_units)
        self.input = np.zeros(num_input + 1)
        self.activation = activation

    #for the moment just one vector at a time
    #TODO incorporate bias in matrix and input vector
    def forward(self, sample): #should work
        self.input = self.prepend_one(sample)
        self.output_before_ac = self.weights.dot(self.input)
        return self.activation.eval(self.output_before_ac)

    #takes error from next layer and calcs and updates weights update
    def backward(self, error, learn_rate, is_output_layer):
        debug1 = self.activation.eval_diff(self.output_before_ac)
        layer_error = self.activation.eval_diff(self.output_before_ac) * error
        if not is_output_layer:
            debug3 = np.sum(self.weights[:, 1:], axis=1)
            layer_error = layer_error * np.sum(self.weights[:, 1:], axis=1) #TODO test wether this is a sum of vectors in right direction

        debug = np.outer(error, self.input)
        debug2 = [learn_rate * x for x in np.asarray(np.outer(error, self.input))]
        self.weights = self.weights + [learn_rate * x for x in np.asarray(np.outer(error, self.input))] #TODO test wether this results in a matrix
        return layer_error

    def prepend_one(self, sample): #works
        return np.array(np.append([1], sample))


class NeuralNet:

    layers: np.array
    cost_function: CostFunction

    #TODO sanitize layer inputs
    def __init__(self, layers, cost_function):
        self.layers = layers
        self.cost_function = cost_function

    def train(self, data, labels, epochs, learning_rate):
        epoch = 0

        while epochs > 0:

            gesamt_error = 0
            epochs -= 1
            out_test = 0
            for i, sample in enumerate(data):
                out = sample
                for j, layer in enumerate(self.layers):
                    out = layer.forward(out)
                loss = np.sum(self.cost_function.function(labels[i], out))
                out_test = out
                error = self.cost_function.function_diff(labels[i], out) #this cant work
                gesamt_error += loss
                is_output_layer = True
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate, is_output_layer)
                    is_output_layer = False
            if epoch % 10 == 0:
                print("Epoche {:d} loss: {:f}".format(epoch, gesamt_error))
            epoch +=1


    def predict(self, data):
        labels = np.empty((data.shape[0]))
        for i, instance in enumerate(data):
            out = instance
            for layer in self.layers:
                out = layer.forward(out)
            labels[i] = out
        return labels


def mse(x, y):
    return 0.5 * np.power(x - y, 2)


def relu(vector):

    return np.array([np.maximum(0, x) for x in vector])


def relu_diff(vector):
    relu_vector = relu(vector)
    for i, x in enumerate(relu_vector):
        if x >= 0:
            relu_vector[i] = 1
        else:
            relu_vector[i] = 0
    return relu_vector

def test():
    #interval: [0,1]
    number_of_samples = 200
    x_values = np.array([np.array([(x / number_of_samples)]) for x in range(number_of_samples)])
    y_values = np.array([np.array([((x / number_of_samples)**2 )]) for x in range(number_of_samples)])

    input_layer = Layer(1, 101, ActivationFunction(relu, relu_diff))
    hidden_1 = Layer(101, 101, ActivationFunction(relu, relu_diff))
    output_layer = Layer(101, 1, ActivationFunction(lambda x: x, lambda x: 1))

    layers = np.array([input_layer, output_layer])

    net = NeuralNet(layers, CostFunction(mse, lambda z, y: z - y))

    net.train(x_values, y_values, 150, 0.01)

    predicted_y = net.predict(x_values)
    plt.plot(x_values, predicted_y)
    plt.plot(x_values, y_values)
    plt.show()

test()


