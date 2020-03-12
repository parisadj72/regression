import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#######################################
####Q2.1: Normalization

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.
    
    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO

    high = 1.0
    low = 0.0

    mins = np.min(train, axis=0)
    maxs = np.max(train, axis=0)
    rng = maxs - mins

    train_normalized = high - (((high - low) * (maxs - train)) / rng)

    mins = np.min(test, axis=0)
    maxs = np.max(test, axis=0)
    rng = maxs - mins

    test_normalized = high - (((high - low) * (maxs - test)) / rng)

    return (train_normalized, test_normalized)


########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    #TODO

    num_instances, num_features = X.shape[0], X.shape[1]
    l = y - X.dot(theta)
    loss = sum(i*i for i in l)/num_instances
    return loss    

########################################

def compute_regularized_loss(theta, lambda_reg):

    loss = lambda_reg * sum(i*i for i in theta)
    return loss

def compute_total_loss(X, y, theta, lambda_reg):
    '''
    compute squared loss + regularized loss
    '''
    num_instances, num_features = X.shape[0], X.shape[1]
    loss = 0
    l = y - X.dot(theta)
    loss = sum(i*i for i in l)/num_instances + lambda_reg * theta.dot(theta)
    return loss

########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO

    num_instances, num_features = X.shape[0], X.shape[1]
    grad = -2*(y-X.dot(theta)).dot(X)/num_instances
    return grad    
       
        
###########################################
###Q2.3a: Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO

    for i in range(num_features):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus[i] -= epsilon
        ag = (compute_square_loss(X, y, theta_plus) - 
        compute_square_loss(X, y, theta_minus))/(2*epsilon)
        approx_grad[i] = ag
        if abs(approx_grad[i] - true_gradient[i]) > tolerance:
            return False
    
    return True
    
#################################################
###Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO

    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)
    
    for i in range(num_features):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus[i] -= epsilon
        ag = (objective_func(X, y, theta_plus) - objective_func(X, y, theta_minus))/(2*epsilon)
        approx_grad[i] = ag
        if abs(approx_grad[i] - true_gradient[i]) > tolerance:
            return False
    
    return True



####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False, stop_if = 0, stop = 0.01):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_gradient - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    #TODO

    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    if check_gradient == True & grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4) == False:
        return False
    for i in range(num_iter):
        if check_gradient == True and not grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
            return False
        gradient = compute_square_loss_gradient(X, y, theta)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
        if stop_if == 1:
            if loss_hist[i+1]-loss_hist[i-1] < stop:
                break
    return theta_hist, loss_hist

####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
    
def batch_grad_descent_back(X, y, beta = 0.4, alpha=0.1, num_iter=1000, check_gradient=False, stop_if = 0, stop = 0.01):
    
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    temp = alpha
    for i in range(num_iter):
        alpha = temp
        gradient = compute_square_loss_gradient(X, y, theta)
        while (compute_square_loss(X, y, theta - alpha * gradient) > compute_square_loss(X, y, theta) - alpha / 2 * np.sum([gr**2 for gr in gradient])):
            alpha *= beta
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
        if stop_if == 1:
            if loss_hist[i+1]-loss_hist[i-1] < stop:
                break
    return theta_hist, loss_hist

###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO

    num_instances, num_features = X.shape[0], X.shape[1]
    grad = -2*((y-X.dot(theta)).dot(X))/num_instances+2*lambda_reg * theta
    return grad

###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    
    #TODO

    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        gradient = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist
    
#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss

def regularized_grad_descent_plot_l(X_train, y_train, X_test, y_test, expos, alpha=0.025, num_iter=1000):

    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    losses_train = np.zeros(len(expos))
    losses_test = np.zeros(len(expos))
    lambda_reg=[pow(10, expo) for expo in expos]
    for i, expo in enumerate(expos): 
        num_instances, num_features = X_train.shape[0], X_train.shape[1]
        theta_hist, loss_hist = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg = lambda_reg[i], num_iter=num_iter)
        theta = theta_hist[num_iter]
        losses_train[i] = compute_square_loss(X_train, y_train, theta)
        losses_test[i] = compute_square_loss(X_test, y_test, theta)
    plt.figure(figsize=(10, 5))
    plt.plot(np.log10(lambda_reg),losses_train,label='Train')
    plt.plot(np.log10(lambda_reg),losses_test,label='Test')
    plt.xlabel("log(lambda_reg)")
    plt.ylabel("squared loss")
    plt.legend()
    plt.savefig('2571.png')
    plt.show()
    return losses_train, losses_test

#############################################
###Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent_wr(X_, y, alpha=0.1, b=2.5, lambda_reg = 1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """    

    #TODO

    X = np.copy(X_)
    X[:,-1] *= b
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        random_ind = np.random.choice(num_instances, replace=True)
        gradient = compute_regularized_square_loss_gradient(X[random_ind, :].reshape(1,X[random_ind, :].shape[0]), y[random_ind], theta, lambda_reg)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist

################################################

def stochastic_grad_descent(X_, y, alpha_=0.025, b=1.6, lambda_reg = pow(10,-1.75), num_iter=2000):

    X = np.copy(X_)
    X[:,-1] *= b
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    losses = np.zeros(num_iter)
    temp_index = 0
    start = 0
    if not isinstance(alpha_,float):
        start = 100
    for i in range(start,num_iter):
        if not isinstance(alpha_,float):
            if (alpha_ == "1/t"):
                alpha = 1/i
            elif (alpha_ == "1/sqrt(t)"):
                alpha =  1/np.sqrt(i)
            elif (alpha_ == "mode3"):
                alpha = 0.01
                alpha =  alpha/(1+alpha*lambda_reg*i)
            else:
                print ("erro")
                return None, None
        else:
            alpha = alpha_
        index = np.array(range(num_instances))
        np.random.shuffle(index)
        for k, j in enumerate(index):
            gradient = compute_regularized_square_loss_gradient(X[j, :].reshape(1,X[j, :].shape[0]), y[j], theta, lambda_reg)
            theta = theta - alpha * gradient
            theta_hist[i-start, j] = theta
            loss = compute_square_loss(X[j, :].reshape(1,X[j, :].shape[0]), y[j], theta)
            if loss > 999999999999:
                print( '{} overflow'.format(alpha_))
                return None, None 
            loss_hist[i-start, j] = loss
    return theta_hist, loss_hist

################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)


def stochastic_grad_descent_decreasing_steps(X_, y, alpha_=0.025, b=2.5, lambda_reg = pow(10,-1.75), power = 1, option = 0, num_iter=1000):
   
    X = np.copy(X_)
    X[:,-1] *= b
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta_hist[0] = theta
    loss_hist[0] = compute_total_loss(X, y, theta, lambda_reg)
    start_step = 100
    for i in range(start_step, num_iter):
        if option == 0:
            alpha = 1/pow(i,power)
        else:
            alpha = alpha_/(1+alpha_*lambda_reg*i)
        random_ind = np.random.choice(num_instances, replace=True)
        gradient = compute_regularized_square_loss_gradient(X[random_ind, :].reshape(1,X[random_ind, :].shape[0]), y[random_ind], theta, lambda_reg)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_total_loss(X, y, theta, lambda_reg)
    plt.figure(figsize=(10, 5))
    plt.plot(range(start_step+1, num_iter+1),np.log10(loss_hist[start_step+1:]),label='Train')
    plt.xlabel("num_iter")
    plt.ylabel("squared loss")
    plt.legend()
    plt.show()

    return theta_hist, loss_hist

def main():
    #Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term

    #TODO

if __name__ == "__main__":
    main()
    
