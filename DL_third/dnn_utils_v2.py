import numpy as np

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    # a = 1 / (1 + np.exp(-z)
    cache = z
    return a,cache

def sigmoid_backward(da,cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dz = da* s * (1-s)
    # print("dz.shape "+str(dz.shape))
    # print("Z.shape " + str(Z.shape))
    assert(dz.shape == Z.shape)

    return dz

def relu(z):
    a = np.maximum(0,z)
    assert(a.shape==z.shape)
    cache = z
    return a,cache
def relu_backward(da,cache):
    z = cache
    dz = np.array(da,copy = True) #just conberting dz to a correct object
    #when z<=0 you should set dz to 0 as well
    dz[z<=0] = 0
    assert(dz.shape == z.shape)
    return dz