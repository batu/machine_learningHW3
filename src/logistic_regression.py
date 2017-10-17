# Source https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/

import math, random, timeit
import numpy as np
e = math.e
alpha = 0.9
lmda = 0.000001
n = 1

dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]
coef = [-0.406605464, 0.852573316, -1.104746259]


#Sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Make a prediction with using a single row of data and predicitons.
def predict(data_row, coefficients):
    yhat = coefficients[0]
    for i in range(len(data_row)-1):
        yhat += coefficients[i + 1] * data_row[i]
    # Logistic regression
    return sigmoid(yhat)


#train[0:-1] features
#train[-1] label
def coefficients_sgd(train: list, l_rate: float, n_epoch: int):

    #start a timer
    start = timeit.default_timer()

    #initilize the coeffifcent vector with random numbers.
    #a value for each feature.
    coef = [random.random() for i in range(len(train[0]))]

    #start the training.
    for epoch in range(n_epoch):

        #keep track of the sum error.
        sum_error = 0
        for training_row in train:
            y = training_row[-1]

            #predcit using the current coefficients and the active row.
            yhat = predict(training_row, coef)
            #figure out the mean square error
            error = y - yhat
            sum_error += (1/2) * error**2

            #The given learning rate function
            l_rate = n * (epoch+1)**(-alpha)
            #l_rate = alpha #this is much better...

            #update the intercept coef
            coef[0] = coef[0] + l_rate * (error * yhat * (1.0 - yhat)) + lmda*np.dot(coef,coef)

            #update the rest of the intercepts
            # this can be done using matrix operations
            for i in range(len(training_row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * (error * yhat * (1.0 - yhat) * training_row[i]) + lmda*np.dot(coef,coef)

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    end = timeit.default_timer()
    print("The training took {} seconds.".format(end - start))
    return coef

l_rate = alpha
n_epoch = 100
coef = coefficients_sgd(dataset, l_rate, n_epoch)
