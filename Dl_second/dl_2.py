import numpy as np
import matplotlib.pyplot as plt
import sklearn
#from testCases import *
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary
from planar_utils import load_planar_dataset
from planar_utils import sigmoid

np.random.seed(1) # set a seed so that the results are consistent
X, Y = load_planar_dataset()
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
#print("111")
#plt.show()

shape_x = X.shape
shape_y = Y.shape
m = Y.shape[1]

print("the shape of X is: "+str(shape_x))
print("the shape of Y is: "+str(shape_y))
print("I have m = %d training examples!"%(m))

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
#plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

def layer_sizes(X,Y):
    """
    :param X:  input dataset of shape (input size,number of examples)
    :param Y: labels of shape (output size,number of examples)
    :return:
        n_x -- the size of input layer
        n_h -- the size of hidden layer
        n_y -- the size of output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)

#X_assess, Y_assess = load_planar_dataset()
#(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
#print("The size of the input layer is: n_x = " + str(n_x))
#print("The size of the hidden layer is: n_h = " + str(n_h))
#print("The size of the output layer is: n_y = " + str(n_y))

def initialize_parameters(n_x,n_h,n_y):
    """

    :param n_x: the size of input layer
    :param n_h: the size of hidden layer
    :param n_y: the size of output laer
    :return:
        w1 -- weight matrix of shape (n_h,n_x)
        b1 -- bias vectors of shape (n_h,1)
        w2 -- weight matrix of shape (n_y,n_h)
        b2 -- bias vector of shape(n_y,1)
    """
    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1)) * 0.01
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1)) * 0.01

    assert(w1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(w2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
    params = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
    }
    return params

#X_assess, Y_assess = load_planar_dataset()
#(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
#parameters = initialize_parameters(n_x,n_h,n_y)
#print("W1 = " + str(parameters["w1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["w2"]))
#print("b2 = " + str(parameters["b2"]))

def forward_propagate(X,params):
    """

    :param X: input data if size(n_x,m)
    :param params: python dictionary containing your parameters (output of initianlization function)
    :return:
         a2 -- the sigmoid output of the secong activation
         cache -- a dictionary containing z1,a1,z2,a2
    """
    # Retrieve each parameters from the dictionary "params"
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    #Implement forward propagate to calculate a2
    z1 = np.dot(w1,X)+b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = sigmoid(z2)

    assert(a2.shape==(1,X.shape[1]))
    cache = {
        "z1":z1,
        "a1":a1,
        "z2":z2,
        "a2":a2
    }
    return a2,cache
#a2,cache = forward_propagate(X_assess,parameters)

#print(np.mean(cache['z1']) ,np.mean(cache['a1']),np.mean(cache['z2']),np.mean(cache['a2']))
##print("z1 = "+str(cache["z1"]))
#print("a1 = "+str(cache["a1"]))
#print("z2 = "+str(cache["z2"]))
#print("a2 = "+str(cache["a2"]))

def compute_cost(a2,Y,params):
    """

    :param a2:  the sigmoid output of the second activation ,of size (1,number of examples)
    :param Y: "true" labels vectors of shape (1,number of examples)
    :param parameters: a numpy dictionary containing z1,a1,z2,a2
    :return: cost -- cross entropycost given equation
    """
    m = Y.shape[1]
    # compute cross entrory cost
    #logprobs = np.multiply(np.log(a2),Y)+np.multiply(np.log(1-a2),(1-Y))
    #cost = -np.sum(logprobs)/m
    logprobs = np.multiply(np.log(a2), Y) + np.multiply(np.log(1 - a2), 1 - Y)
    cost = -np.sum(logprobs) / m

    cost = np.squeeze(cost)
    # make sure cost is the dimension we expect
    assert(isinstance(cost,float))
    return cost

def backward_propagate(X,Y,parameters,cache):
    """

    :param X: input data of shape (2, number of examples)
    :param Y: "true" labels vector of shape (1, number of examples)
    :param paramaters: python dictionary containing our parameters
    :param cache: a dictionary containing "Z1", "A1", "Z2" and "A2".
    :return: grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    # first ,retrieve w1,w2 from the dictionary "parameters"
    w1 = parameters["w1"]
    w2 = parameters["w2"]

    # retrieve z1,z2 from cache
    a1 = cache["a1"]
    a2 = cache["a2"]

    dz2 = a2 - Y
    dw2 = 1./m*np.dot(dz2,a1.T)
    db2 = 1./m*np.sum(dz2,axis = 1,keepdims = True)

    #dz1 = np.dot(w2.T,dz2)*(1 - np.power(a1,2))
    dz1 = np.multiply(np.dot(w2.T,dz2),(1 - np.power(a1,2)))
    dw1 = 1./m*np.dot(dz1,X.T)
    db1 = 1./m*np.sum(dz1,axis = 1,keepdims=True)

    grads = {
        "dw1":dw1,
        "db1":db1,
        "dw2":dw2,
        "db2":db2
    }
    return grads

def update_parameters(grads,parameters,learning_rate = 1.2):
    """
    :param grads: python dictionary containing your grads (dw1,db1,dw2,db2)
    :param parameters: python dictionary containing your parameters
    :param learning_rate:
    :return: parameters -- python dictionary containing your updated parameters
    """
    #retrieve w1,b1,w2,b2 from parameters
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    #retrieve dw1,db1,dw2,db2 from grads
    dw1 = grads["dw1"]
    db1 = grads["db1"]
    dw2 = grads["dw2"]
    db2 = grads["db2"]

    #update parameter

    w1 = w1 - learning_rate*dw1
    b1 = b1 - learning_rate*db1
    w2 = w2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2

    parameters = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
    }
    return parameters

def nn_model(X,Y,n_h,num_iterations = 1000, print_cost = False):
    """
    :param X:
    :param Y:
    :param n_h:
    :param num_iterations:
    :param print_cost:
    :return: parameters -- params learnt by the model. They can then used to predict.
    """
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    #initailize parameters ,then retrieve w1,b1,w2,b2. inputs:"n_x,n_h,n_y"
    params = initialize_parameters(n_x,n_h,n_y)
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    #loop (gradient descent)
    for i in range(0,num_iterations):
        # forward_propagate
        a2,cache = forward_propagate(X,params)

        #cost function
        cost = compute_cost(a2,Y,params)

        #backward propagate
        grads = backward_propagate(X,Y,params,cache)

        #update parameter
        params = update_parameters(grads,params)

        #print the cost every 1000 iterations
        if print_cost and i%1000 == 0:
            print("cost after iteration %i: %f"%(i,cost))
    return params

def predict(parameters,X):
    """
    :param parameters: a numpy dictionary containing w1,b1,w2,b2
    :param X: input data of size (n_x,m)
    :return: prediction vector of our model (red:0 / blue:1)
    """
    a2,cache = forward_propagate(X,parameters)
    prediction = (a2>0.5)
    return prediction

# bulid a model with a n_h dimensional hidden layer
params = nn_model(X,Y,n_h = 4,num_iterations=10000,print_cost = True)

# plot the decision boundry
plot_decision_boundary(lambda x:predict(params,x.T),X,Y)
plt.title("Decision boundary for hidden layer size "+ str(4))
plt.show()

predictions = predict(params, X)
print(predictions.shape)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')





