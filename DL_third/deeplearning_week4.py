import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import time
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
# Example of a picture
# index = 7
# plt.imshow(train_x_orig[index])
# plt.show()
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# reshape the train and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

# standardize data to have feature values between 0 and 1
train_x = train_x_flatten/255
test_x = test_x_flatten/255

print("train_x'shape "+str(train_x.shape))
print("test_x' shape "+str(test_x.shape))

layers_dims = [12288,7,1] #  -layer model
layers_dims1 = [12288,20,7,5,1] #  5-layer model

# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)

def two_layer_model(X,Y,layers_dims,learning_rate = 0.0075,num_iterations = 3000,print_cost=False):
    np.random.seed(3)
    grads = {}
    costs = []
    m = X.shape[1]
    # (layers_dims[0],layers_dims[1],layers_dims[2]) = layers_dims
    parameters = initialize_parameters(layers_dims[0],layers_dims[1],layers_dims[2])

    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    for i in range(0,num_iterations):
        # forward propagation
        a1,cache1 = linear_activation_forward(X,w1,b1,activation = "relu")
        a2,cache2 = linear_activation_forward(a1,w2,b2,activation = "sigmoid")
        # print("AL = " + str(a2))
        # compute cost
        cost = compute_cost(a2,Y)

        #initialize backward propagation
        da2 = -(np.divide(Y,a2) - np.divide(1-Y,1-a2))

        # backward propagation
        da1,dw2,db2 = linear_activation_backward(da2,cache2,activation = "sigmoid")
        da0,dw1,db1 = linear_activation_backward(da1,cache1,activation = "relu")

        # set grads
        grads["dw1"] = dw1
        grads["db1"] = db1
        grads["dw2"] = dw2
        grads["db2"] = db2

        # update parameters
        parameters = update_parameters(parameters,grads,learning_rate)

        #retrieve w1,b1,w2,b2
        w1 = parameters["w1"]
        b1 = parameters["b1"]
        w2 = parameters["w2"]
        b2 = parameters["b2"]
        #print the cost every 1000 training eample
        if print_cost and i%100 == 0 :
            costs.append(cost)
            print("cost after iteration {}: {}".format(i,np.squeeze(cost)))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# predictions_train = predict(train_x, train_y, parameters)

def L_layer_model(X,Y,layers_dims,learning_rate = 0.0075,num_iterations = 3000,print_cost = False):
    np.random.seed(3)
    costs = []
    # grads = {}

    # initialize parameter
    parameters = initialize_parameters_deep(layers_dims)
    # print ("W1 = "+ str(parameters["w1"].shape))
    # print ("b1 = "+ str(parameters["b1"].shape))
    # print ("W2 = "+ str(parameters["w2"].shape))
    # print ("b2 = "+ str(parameters["b2"].shape))
    #loop
    print("length parameters "+str(len(parameters)))
    print("length parameters " + str(len(parameters)//2))


    for i in range(0,num_iterations):

        # forward propagation
        AL ,caches = L_model_forward(X,parameters)
        # print("AL = "+str(AL))

        # compute cost
        cost = compute_cost(AL,Y)

        #backward propagation
        grads = L_model_backward(AL,Y,caches)

        #update parameter
        parameters = update_parameters(parameters,grads,learning_rate)

        # print the cost every 100 training example
        if print_cost and i%100 == 0:
            costs.append(cost)
            print("cost after iteration {}: {}".format(i,np.squeeze(cost)))
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# parameters = two_layer_model(train_x,train_y,layers_dims,num_iterations=2500,print_cost=True)

parameters = L_layer_model(train_x, train_y, layers_dims1, num_iterations = 2500, print_cost = True)
Y_prediction_train = predict(train_x, train_y, parameters)
Y_prediction_test = predict(test_x, test_y, parameters)
print("train accuraty: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
print("test accuraty: {}%".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))

print("ahahhahha")
# parameters1 = L_layer_model(train_x, train_y, layers_dims1, num_iterations = 2500, print_cost = True)
# parameters = two_layer_model(train_x,train_y,layers_dims=(n_x,n_h,n_y),num_iterations=2500,print_cost=True)

my_image = "my_img.jpg"   # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")