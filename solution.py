import numpy as np
import matplotlib.pyplot as plt

def magic_function(x): # do not change this line!
    # implement a function that is suitable for the use in main.py
    # x is a N-by-D numpy array
    return x


def f(x,w,b): # do not change this line!
    # implement the function f() here
    # x is a N-by-D numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # Should return three things:
    # 1) the output of the f function, as a N dimensional numpy array,
    # 2) gradient of f with respect to w,
    # 3) gradient of f with respect to b
    x=x.T
    N = x.shape[1]
    Y_prediction = np.zeros((N))
    w = w.reshape(x.shape[0], 1)
    B=np.dot(w.T,x)+b
    A = sigmoid(B.T) # compute activation
    for i in range(A.shape[0]):
        if (A[i] <= 0.5):
            Y_prediction[i] = 0
        else:
            Y_prediction[i] = 1
    return Y_prediction
    
def sigmoid(z):
    alpha = 0.001
    Y_prediction = np.zeros((z.shape[0]))
    for i in range(z.shape[0]):
        if (z[i] < 0):
            Y_prediction[i] = -alpha*z[i]
        elif(0 <= z[i] <= 1):
             Y_prediction[i] = 3*pow(z[i],3)-4*pow(z[i],2)+2*z[i]
        elif (z[i] > 1):
             Y_prediction[i] = z[i]
    return Y_prediction

def loss(x,y,w,b): # do not change this line!
    # implement the loss function here
    # x is a N-by-D numpy array
    # y is a N dimensional numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # Should return three items:
    # 1) the loss which is a scalar,
    # 2) the gradient of the loss with respect to w,
    # 3) the gradient of the loss with respect to b
    N = x.shape[1]
    A = sigmoid(np.dot(w.T,x)+b) # compute activation
    cost = (1/N)*np.sum((y -A)) ** 2
    db = -(2/N) * np.sum((y -A))
    dw = -(2/N) * np.dot(x,(y -A))
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,"db": db}
    return grads, cost

def minimize_loss(x,y,w,b,params): # do not change this line!
    # implement an optimizer for w and b here
    # x is a N-by-D numpy array
    # y is a N dimensional numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # params is a list contaning hyper parameters for optimizer
    # Should return the final values of w and b
    costs = []
    costfunction = []
    learning_rate = 0.005
    tolerance = 0.10
    x=x.T
    y=y.T
    grads, cost = loss(x, y, w, b)
    i = 0
    
    # Nesterov Accelerated Gradient Descent Equations
    #-------------------------------------------------
    a = 0
    nesterov_Momentum1 = np.zeros(w.shape[0])
    nesterov_Momentum3 = 0
    #-------------------------------------------------
    while cost > tolerance:
        grads, cost = loss(x, y, w, b)
        dw = grads["dw"]
        db = grads["db"]
        
        #Gradient Descent Update
        #-------------------------------------------------
        #w = w-(learning_rate*dw)
        #b = b-(learning_rate*db)
        #-------------------------------------------------

        
        #Nesterov Accelerated Gradient Descent Update
        #-------------------------------------------------
        a = 0.5 *(1 + np.sqrt(1 + 4*pow(a,2)));      
        a2 = 0.5*(1 + np.sqrt(1 + 4*pow(a,2)));       
        gamma=(1-a)/a2;
        nesterov_Momentum2 = w-(learning_rate*dw)
        w = (1-gamma)*nesterov_Momentum2 + gamma*nesterov_Momentum1
        nesterov_Momentum4 = b-(learning_rate*db)
        b = (1-gamma)*nesterov_Momentum4 + gamma*nesterov_Momentum3
        nesterov_Momentum1 = nesterov_Momentum2
        nesterov_Momentum3 = nesterov_Momentum4
        #-------------------------------------------------
        i = i + 1
        costfunction.append(cost)
        print ("Cost after iteration %i: %f" %(i, cost))
        
    plt.plot(costfunction)
    plt.ylabel('Cost function')
    plt.xlabel('Iterations')
    plt.title("Learning rate = 0.0005, alpha = 0.0001")
    plt.show()
    return w,b
