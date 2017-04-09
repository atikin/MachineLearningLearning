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


#todo make this an interface
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
        self.weights = np.random.normal(0, 0.5, (num_hidden_units, num_input + 1))
        self.output_before_ac = np.zeros(num_hidden_units + 1)
        self.input = np.zeros(num_input + 1)
        self.activation = activation

    #for the moment just one vector at a time
    def forward(self, sample):
        self.input = self.prepend_one(sample)
        self.output_before_ac = self.weights.dot(self.input)
        return self.activation.eval(self.output_before_ac)

    #takes error from next layer and calcs and updates weights update
    def backward(self, error, learn_rate):
        layer_error = self.activation.eval_diff(self.output_before_ac) * error
        layer_error_new = self.weights[:, 1:].T.dot(layer_error)
        self.weights = self.weights + [learn_rate * x for x in np.asarray(np.outer(layer_error, self.input))]
        return layer_error_new

    def prepend_one(self, sample):
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

            permutation = np.random.permutation(data.shape[0])
            shuffled_data = np.array([data[i] for i in permutation])
            shuffled_labels = np.array([labels[i] for i in permutation])
            gesamt_error = 0
            epochs -= 1
            for i, sample in enumerate(shuffled_data):
                out = sample
                for j, layer in enumerate(self.layers):
                    out = layer.forward(out)

                loss = np.sum(self.cost_function.function(shuffled_labels[i], out))
                gesamt_error += loss

                error = self.cost_function.function_diff(shuffled_labels[i], out)

                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
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
    #interval: [-1,1]
    number_of_samples = 100
    x_values = np.array([np.array([(x / number_of_samples)]) for x in range(-number_of_samples, number_of_samples)])
    y_values = np.array([np.array([np.sin(np.pi * x / number_of_samples)]) for x in range(-number_of_samples, number_of_samples)])

    input_layer = Layer(1, 20, ActivationFunction(relu, relu_diff))
    hidden_1 = Layer(20, 20, ActivationFunction(relu, relu_diff))
    hidden_2 = Layer(4, 1, ActivationFunction(relu, relu_diff))
    output_layer = Layer(20, 1, ActivationFunction(lambda x: x, lambda x: 1))

    layers = np.array([input_layer, hidden_1,  output_layer])

    net = NeuralNet(layers, CostFunction(mse, lambda z, y: z - y))

    net.train(x_values, y_values, 350, 0.001)

    predicted_y = net.predict(x_values)
    plt.plot(x_values, predicted_y)
    plt.plot(x_values, y_values)
    plt.show()

test()


