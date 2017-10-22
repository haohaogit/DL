import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

from dnn_app_utils_v3 import L_model_forward,L_model_backward,cost_function,update_parameters,predict

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()

def initialize_parameters_zeros(layers_dims):
    np.random.seed(3)
    L = len(layers_dims)
    parameters = {}
    for i in range(1,L):
        parameters["w"+str(i)] = np.zeros((layers_dims[i],layers_dims[i-1]))
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))
    return parameters

def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    L = len(layers_dims)
    parameters = {}
    for i in range(1,L):
        parameters["w"+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))
    return parameters

parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["w1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["w2"]))
print("b2 = " + str(parameters["b2"]))

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    L = len(layers_dims)
    parameters = {}
    for i in range(1,L):
        parameters["w"+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1]) * np.sqrt(2/layers_dims[i-1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))
    return parameters
# parameters = initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["w1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["w2"]))
# print("b2 = " + str(parameters["b2"]))

def model(X,Y,learning_rate = 0.01,num_iteration = 15000,print_coat=False,initialization="he"):
    grads = {}
    costs = []
    L = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]
    parameters = {}

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(0,num_iteration):
        AL ,caches = L_model_forward(parameters,X)

        cost = cost_function(AL,Y)

        grads = L_model_backward(AL,Y,caches)

        parameters = update_parameters(parameters,grads,learning_rate)

        if i%1000 == 0 and print_coat:
            costs.append(cost)
            print("cost after iteration {}: {}".format(i, np.squeeze(cost)))
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

parameters = model(train_X, train_Y, initialization = "random",print_coat=True)
print ("On the train set:")
predictions_train = predict(train_X, parameters)
print ("On the test set:")
predictions_test = predict(test_X, parameters)
print("train accuraty: {} %".format(100 - np.mean(np.abs(predictions_train - train_Y)) * 100))
print("test accuraty: {}%".format(100 - np.mean(np.abs(predictions_test - test_Y)) * 100))

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict( x.T,parameters), train_X, train_Y)
