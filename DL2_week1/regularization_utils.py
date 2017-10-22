import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io

from testCases import compute_cost_with_regularization_test_case,backward_propagation_with_regularization_test_case
from dnn_app_utils_v3 import L_model_forward,L_model_backward,cost_function,update_parameters,predict,L_model_backward_regularization
from dnn_app_utils_v3 import linear_forward
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward,relu_backward_keepprob
from testCases import forward_propagation_with_dropout_test_case

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)

    return train_X, train_Y, test_X, test_Y


def initialize_parameters(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for i in range(1,L):
        parameters["w"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
    return parameters

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost_temp = np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL))
    cost = -np.sum(cost_temp,axis=1,keepdims=True)/m
    cost = np.squeeze(cost)
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]

    cross_entropy_cost = compute_cost(AL, Y)  # This gives you the cross-entropy part of the cost
    L = len(parameters)//2
    L2_regularization_cost= 0.
    ### START CODE HERE ### (approx. 1 line)
    for i in range(1,L):
        L2_regularization_cost =  L2_regularization_cost+(np.sum(np.square(parameters["w"+str(i)])))
    ### END CODER HERE ###

    cost = cross_entropy_cost + (L2_regularization_cost * lambd / (2 * m))
    return cost

def compute_cost_with_regularization1(AL, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["w1"]
    W2 = parameters["w2"]
    W3 = parameters["w3"]

    cross_entropy_cost = compute_cost(AL, Y)  # This gives you the cross-entropy part of the cost

    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * lambd / (2 * m)
    ### END CODER HERE ###

    cost = cross_entropy_cost + L2_regularization_cost
    return cost
# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
# print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))

# X_assess, Y_assess, caches = backward_propagation_with_regularization_test_case()
# linear_cache, activation_cache = caches[2]
# grads = L_model_backward_regularization(activation_cache, Y_assess, caches, lambd=0.7)
# print("dW1 = " + str(grads["dw1"]))
# print("dW2 = " + str(grads["dw2"]))
# print("dW3 = " + str(grads["dw3"]))
#
# train_X, train_Y, test_X, test_Y = load_2D_dataset()
def linear_activity_forward_keepprob(A_prev,w,b,activation,keep_prob):
    if("sigmoid" == activation):
        z,linear_cache = linear_forward(w,b,A_prev)
        A ,activation_cache = sigmoid(z)
        d1 = A


    if("relu" == activation):
        z,linear_cache = linear_forward(w,b,A_prev)
        A ,activation_cache = relu(z)
        d1 = np.random.rand(A.shape[0], A.shape[1])
        d1 = (d1 < keep_prob)
        A = A * d1
        A = A / keep_prob
        # print("d1 shape"+str(d1.shape))
        # print("A shape" + str(A.shape))
    cache = (linear_cache, activation_cache)
    return A,cache,d1



def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    np.random.seed(1)

    L = len(parameters) // 2
    caches = []
    d1s = []
    A = X
    for i in range(1, L):
        A_prev = A
        A, cache ,d1= linear_activity_forward_keepprob(A_prev, parameters["w" + str(i)], parameters["b" + str(i)], "relu",keep_prob)
        caches.append(cache)
        d1s.append(d1)
    AL, cache ,d1 = linear_activity_forward_keepprob(A, parameters["w" + str(L)], parameters["b" + str(L)], "sigmoid",keep_prob)
    assert (AL.shape == (1, X.shape[1]))
    caches.append(cache)
    d1s.append(d1)
    return AL, caches,d1s

# X_assess, parameters = forward_propagation_with_dropout_test_case()
# A3, cache ,d1s= forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
# print ("A3 = " + str(A3))

def linear_backward_keepprob(dz,d1,cache,keep_prob):
    A_prev,w,b = cache
    m = A_prev.shape[1]

    dw = np.dot(dz,A_prev.T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dA_prev = np.dot(w.T,dz)

    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev,dw,db
def linear_backward(dz,cache):
    A_prev,w,b = cache
    m = A_prev.shape[1]

    dw = np.dot(dz,A_prev.T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dA_prev = np.dot(w.T,dz)

    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev,dw,db

def linear_activation_backward_keepprob(dA,d1,cache,activation,keep_prob):
    linear_cache,activation_cache = cache
    if("sigmoid" == activation):
        dz = sigmoid_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward(dz,linear_cache)
    if("relu"==activation):
        dz = relu_backward_keepprob(dA,d1,activation_cache,keep_prob)
        dA_prev, dw, db = linear_backward(dz, linear_cache)
    return dA_prev,dw,db



def backward_propagation_with_dropout(AL, Y, caches,d1s, keep_prob):
    L = len(caches)
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    d1 = d1s[L-2]
    grads["dA" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward_keepprob(dAL, d1,current_cache,
                                                                                                  "sigmoid",keep_prob)

    for i in reversed(range(L - 1)):
        # print("i :"+str(i))
        current_cache = caches[i]
        d1 = d1s[i]
        dA_prev_temp, dw, db = linear_activation_backward_keepprob(grads["dA" + str(i + 2)], d1,current_cache, "relu",keep_prob)
        grads["dA" + str(i + 1)] = dA_prev_temp
        grads["dw" + str(i + 1)] = dw
        grads["db" + str(i + 1)] = db

    return grads


def compute_cost(a3, Y):
    """
    Implement the cost function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]

    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1. / m * np.nansum(logprobs)

    return cost