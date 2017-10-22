import numpy as np
import h5py
import matplotlib.pyplot as plt

from dnn_utils_v2 import sigmoid,sigmoid_backward, relu, relu_backward

# plt.rcParams['figure.figsize'] = (5.0, 4.0) #set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1) #使得随机函数的调用具有一致性

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)

    w1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    w2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    assert(w1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(w2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))

    parameters = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
    }
    return parameters
# parameters = initialize_parameters(2,2,2)
# print("w1= "+str(parameters["w1"]))
# print("b1= "+str(parameters["b1"]))
# print("w2= "+str(parameters["w2"]))
# print("b2= "+str(parameters["b2"]))
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters
# def initialize_parameters_deep(layer_dims):
#     np.random.seed(3)
#     L = len(layer_dims)
#     parameters = {}
#
#     for i in range(1,L):
#
#         parameters["w"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
#         parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
#
#         assert(parameters["w"+str(i)].shape == (layer_dims[i],layer_dims[i-1]))
#         assert(parameters["b"+str(i)].shape == (layer_dims[i],1))
#     return parameters
# parameters = initialize_parameters_deep([5,4,3])
# print("w1= "+str(parameters["w1"]))
# print("b1= "+str(parameters["b1"]))
# print("w2= "+str(parameters["w2"]))
# print("b2= "+str(parameters["b2"]))

def linear_forward(A,w,b):
    # Z = np.dot(w,A) + b
    Z = np.dot(w, A) + b
    # assert(Z.shape == (w.shape[0],A.shape[1]))
    cache = (A ,w ,b)
    return Z,cache

def linear_activation_forward(A_prev,w,b,activation):

    if activation == "sigmoid":
        z ,linear_cache = linear_forward(A_prev,w,b)
        A ,activation_cache = sigmoid(z)

    elif activation == "relu":
        z ,linear_cache = linear_forward(A_prev,w,b)
        A ,activation_cache = relu(z)

    assert(A.shape == (w.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    # print("A = " + str(A))
    return A, cache

# def linear_activation_forward_test_case():
#     """
#     X = np.array([[-1.02387576, 1.12397796],
#  [-1.62328545, 0.64667545],
#  [-1.74314104, -0.59664964]])
#     W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
#     b = 5
#     """
#     np.random.seed(2)
#     A_prev = np.random.randn(3,2)
#     W = np.random.randn(1,3)
#     b = np.random.randn(1,1)
#     return A_prev, W, b
# A_prev, W, b = linear_activation_forward_test_case()
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))
#

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))



def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
        ### END CODE HERE ###

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    ### END CODE HERE ###

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches

def L_model_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
        [-1.07296862,  0.86540763, -2.3015387 ]]),
 'W2': np.array([[ 1.74481176, -0.7612069 ]]),
 'b1': np.array([[ 0.],
        [ 0.]]),
 'b2': np.array([[ 0.]])}
    """
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"w1": W1,
                  "b1": b1,
                  "w2": W2,
                  "b2": b2}

    return X, parameters


# X, parameters = L_model_forward_test_case()
# # print("x= "+str(X))
# # print("w1= "+str(parameters["w1"]))
# # print("b1= "+str(parameters["b1"]))
# # print("w2= "+str(parameters["w2"]))
# # print("b2= "+str(parameters["b2"]))
# # print(str(parameters["w1"].shape[0]))
# # print(str(X.shape[1]))
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

def compute_cost(AL,Y):
    m = AL.shape[1]
                    # np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)),axis = 1,keepdims=True)/m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost
def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])

    return Y, aL


# Y, AL = compute_cost_test_case()
#
# print("cost = " + str(compute_cost(AL, Y)))
def linear_backward(dz,cache):
    A_prev,w,b = cache
    m = A_prev.shape[1]

    dw = 1/m*np.dot(dz,A_prev.T)
    db = 1/m*np.sum(dz,axis = 1,keepdims = True)
    dA_prev = np.dot(w.T,dz)

    assert(dA_prev.shape == A_prev.shape)
    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)

    return dA_prev,dw,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache

    if activation == "sigmoid":
        dz = sigmoid_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward(dz,linear_cache)
    elif activation == "relu":
        dz = relu_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward(dz,linear_cache)
    return dA_prev,dw,db
def linear_activation_backward_test_case():
    """
    aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
    """
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache


# AL, linear_activation_cache = linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line , Y is the same shape as AL

    #initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dw"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

    for i in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[i]
        dA_prev_temp,dw_temp,db_temp = linear_activation_backward(grads["dA"+str(i+2)],current_cache,"relu")
        grads["dA"+str(i+1)] = dA_prev_temp
        grads["dw"+str(i+1)] = dw_temp
        grads["db"+str(i+1)] = db_temp
    return grads



def L_model_backward_test_case():
    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    w1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, w1, b1), Z1)

    A2 = np.random.randn(3,2)
    w2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ( (A2, w2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches
# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dw1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        parameters["w" + str(i + 1)] = parameters["w" + str(i + 1)] - learning_rate * grads["dw" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
    return parameters




def update_parameters_test_case():
    """
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
        [-1.8634927 , -0.2773882 , -0.35475898],
        [-0.08274148, -0.62700068, -0.04381817],
        [-0.47721803, -1.31386475,  0.88462238]]),
 'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
        [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
        [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
 'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
        [-0.16051336, -0.76883635, -0.23003072]]),
 'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
 'b2': np.array([[ 0.],
        [ 0.],
        [ 0.]]),
 'b3': np.array([[ 0.],
        [ 0.]])}
    grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]),
 'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ]]),
 'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
 'da1': np.array([[ 0.70760786,  0.65063504],
        [ 0.17268975,  0.15878569],
        [ 0.03817582,  0.03510211]]),
 'da2': np.array([[ 0.39561478,  0.36376198],
        [ 0.7674101 ,  0.70562233],
        [ 0.0224596 ,  0.02065127],
        [-0.18165561, -0.16702967]]),
 'da3': np.array([[ 0.44888991,  0.41274769],
        [ 0.31261975,  0.28744927],
        [-0.27414557, -0.25207283]]),
 'db1': 0.75937676204411464,
 'db2': 0.86163759922811056,
 'db3': -0.84161956022334572}
    """
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"w1": W1,
                  "b1": b1,
                  "w2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dw1": dW1,
             "db1": db1,
             "dw2": dW2,
             "db2": db2}

    return parameters, grads
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print ("W1 = "+ str(parameters["w1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["w2"]))
# print ("b2 = "+ str(parameters["b2"]))
def predict(X,Y,parameters):
    AL, caches = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    return predictions










