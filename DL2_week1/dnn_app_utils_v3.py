import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward
from lr_utils import load_dataset

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

# standardize data to have feature values between 0 and 1
train_x = train_x_flatten/255
test_x = test_x_flatten/255

# print("train_x'shape "+str(train_x.shape))
# print("test_x' shape "+str(test_x.shape))
def initialize_L_layer(dims_layers):
    np.random.seed(3)
    parameters = {}
    L = len(dims_layers)
    for i in range(1,L):
        parameters["w"+str(i)] = np.random.randn(dims_layers[i],dims_layers[i-1])*0.3
        parameters["b"+str(i)] = np.zeros((dims_layers[i], 1))

        assert (parameters["w"+str(i)].shape==(dims_layers[i],dims_layers[i-1]))
        assert( parameters["b"+str(i)].shape==(dims_layers[i],1))
    return parameters
dims_layers = [12288,20,7,5,1]
# parameters = initialize_L_layer([5,4,3])
# print("w1= "+str(parameters["w1"]))
# print("b1= "+str(parameters["b1"]))
# print("w2= "+str(parameters["w2"]))
# print("b2= "+str(parameters["b2"]))
def linear_forward(w,b,A):
    z = np.dot(w,A)+b
    cache = (A,w,b)
    return z,cache


def linear_activity_forward(A_prev,w,b,activation):
    if("sigmoid" == activation):
        z,linear_cache = linear_forward(w,b,A_prev)
        A ,activation_cache = sigmoid(z)
    if("relu" == activation):
        z,linear_cache = linear_forward(w,b,A_prev)
        A ,activation_cache = relu(z)
    cache = (linear_cache,activation_cache)
    return A,cache

def L_model_forward(parameters,X):

    L = len(parameters) // 2
    caches = []
    A = X
    for i in range(1,L):
        A_prev = A
        A,cache = linear_activity_forward(A_prev,parameters["w"+str(i)],parameters["b"+str(i)],"relu")
        caches.append(cache)
    AL,cache = linear_activity_forward(A,parameters["w"+str(L)],parameters["b"+str(L)],"sigmoid")
    assert(AL.shape == (1,X.shape[1]))
    caches.append(cache)
    return AL,caches



def cost_function(AL,Y):
    m = Y.shape[1]
    # cost_temp = np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL))
    # cost = -np.sum(cost_temp,axis = 1,keepdims=True)/m
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)), axis=1, keepdims=True) / m
    cost = np.squeeze(cost)
    return cost

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

def linear_backward_regularization(dz,cache,lambd):
    A_prev,w,b = cache
    m = A_prev.shape[1]

    dw = np.dot(dz,A_prev.T)/m + lambd*w/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dA_prev = np.dot(w.T,dz)

    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev,dw,db


def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    if("sigmoid" == activation):
        dz = sigmoid_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward(dz,linear_cache)
    if("relu"==activation):
        dz = relu_backward(dA,activation_cache)
        dA_prev, dw, db = linear_backward(dz, linear_cache)
    return dA_prev,dw,db

def linear_activation_backward_regularization(dA,cache,activation,lambd):
    linear_cache,activation_cache = cache
    if("sigmoid" == activation):
        dz = sigmoid_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward_regularization(dz,linear_cache,lambd)
    if("relu"==activation):
        dz = relu_backward(dA,activation_cache)
        dA_prev, dw, db = linear_backward_regularization(dz, linear_cache,lambd)
    return dA_prev,dw,db


def L_model_backward(AL,Y,caches):
    L = len(caches)
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]

    grads["dA" + str(L)],grads["dw"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp,dw,db = linear_activation_backward(grads["dA"+str(i+2)],current_cache,"relu")
        grads["dA"+str(i+1)] = dA_prev_temp
        grads["dw"+str(i+1)] = dw
        grads["db"+str(i+1)] = db

    return grads

def L_model_backward_regularization(AL,Y,caches,lambd):
    L = len(caches)
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]

    grads["dA" + str(L)],grads["dw"+str(L)],grads["db"+str(L)] = linear_activation_backward_regularization(dAL,current_cache,"sigmoid",lambd)

    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp,dw,db = linear_activation_backward_regularization(grads["dA"+str(i+2)],current_cache,"relu",lambd)
        grads["dA"+str(i+1)] = dA_prev_temp
        grads["dw"+str(i+1)] = dw
        grads["db"+str(i+1)] = db

    return grads


def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for i in range(L):
        parameters["w" + str(i + 1)] = parameters["w" + str(i + 1)] - learning_rate * grads["dw" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
    return parameters

# def L_layer_model(X,Y,dims_layers,learning_rate = 0.0075,num_iterations = 2500,print_cost = False):
#     np.random.seed(3)
#     parameters = initialize_L_layer(dims_layers)
#     costs = []
#     for i in range(1,num_iterations):
#         AL,caches = L_model_forward(parameters,X)
#         cost = cost_function(AL,Y)
#
#         grads = L_model_backward(AL,Y,caches)
#         update_parameters(parameters,grads,learning_rate)
#
#         if i%100 == 0 and print_cost:
#             costs.append(cost)
#             print("cost after iteration {}: {}".format(i,np.squeeze(cost)))
#
#     plt.plot(costs)
#     plt.ylabel("cost")
#     plt.xlabel("iterations (per tens)")
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
#     return parameters

def predict(X,parameters):
    AL, caches = L_model_forward(parameters,X)
    predictions = (AL > 0.5)
    return predictions


# parameters = L_layer_model(train_x, train_y, dims_layers,learning_rate=0.0095, num_iterations = 3000, print_cost = True)
# Y_prediction_train = predict(train_x, train_y, parameters)
# Y_prediction_test = predict(test_x, test_y, parameters)
# print("train accuraty: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
# print("test accuraty: {}%".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))








