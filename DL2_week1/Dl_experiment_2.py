import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

from dnn_app_utils_v3 import L_model_forward,L_model_backward,cost_function,update_parameters,predict,L_model_backward_regularization
# from dnn_app_utils_v3 import
from regularization_utils import load_2D_dataset,initialize_parameters,forward_propagation_with_dropout,backward_propagation_with_dropout,compute_cost_with_regularization
from regularization_utils import compute_cost
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

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):

    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            # a3, cache = forward_propagation(X, parameters)
            AL, caches = L_model_forward(parameters, X)
        elif keep_prob < 1:
            AL, caches,d1s = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            # cost = compute_cost(a3, Y)
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)

        # Backward propagation.
        assert(lambd==0 or keep_prob==1)# it is possible to use both L2 regularization and dropout,
                                         # but this assignment will only explore one at a time，所以这个语句是要确保一个失效，不同时使用
        #如果想要同时都支持，需要设置一个新的函数
        if lambd == 0 and keep_prob == 1:
            # grads = backward_propagation(X, Y, cache)
            grads = L_model_backward(AL, Y, caches)

        elif lambd != 0:
            # grads = backward_propagation_with_regularization(X, Y, cache, lambd)
            grads = L_model_backward_regularization(AL, Y, caches,lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(AL, Y, caches, d1s,keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
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


# parameters = model(train_X, train_Y, lambd = 0.7)
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
# parameters = model(train_X, train_Y)
#(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1
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

