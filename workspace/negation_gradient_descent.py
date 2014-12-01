from numpy import *
from numpy.random import *

def gradientDescent(F, dF, d):
    w = zeros((d, 1))
    numIters = 1000
    eta = 0.01  # step size
    for t in range(numIters):
        value = F(w)
        gradient = dF(w)
        print "t = %s, w = %s" % (t, w)
        w = w - eta * gradient
    return w

def stochasticGradientDescent(sF, sdF, d, n):
    w = zeros((d, 1))
    numIters = 1000
    eta = 0.01  # step size
    for t in range(numIters):
        print "t = %s, w = %s" % (t, w)
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            w = w - eta * gradient
    return w

############################################################

# Generate artificial data
# true_w = array([1, 2, 3, 4, 5])
# d = len(true_w)
# numExamples = 10000
# def generateExample():
#     x = randn(d)
#     y = true_w.dot(x) + randn()
#     return (x, y)
# points = [generateExample() for _ in range(numExamples)]

## create map from word to count
counts = {}
# with open('vocab.txt', 'r') as countF:
#     for line in countF:
#         line = line.split()
#         counts[line[0]] = int(line[1])
#print counts


#trainX = [[float(x) if i else x for i, x in enumerate(line.split())] for line in open('originalVectorsTrain.txt', 'r')]
#trainY = [[float(x) if i else x for i, x in enumerate(line.split())] for line in open('negationVectorsTrain.txt', 'r')]
#testX = [[float(x) if i else x for i, x in enumerate(line.split())] for line in open('originalVectorsTest.txt', 'r')]
#testY = [[float(x) if i else x for i, x in enumerate(line.split())] for line in open('negationVectorsTest.txt', 'r')]

Vnot = array([float(x) for i, x in enumerate(open('not.txt', 'r').readline().split()) if i])
d = Vnot.shape[0]
Vnot = Vnot.reshape((d, 1))

trainX = [array([float(x) for  x in line.split()[1:]]).reshape((d, 1)) for line in open('originalVectorsTrain.txt', 'r')]
trainY = [array([float(x) for  x in line.split()[1:]]).reshape((d, 1)) for line in open('negationVectorsTrain.txt', 'r')]
testX = [array([float(x) for  x in line.split()[1:]]).reshape((d, 1)) for line in open('originalVectorsTest.txt', 'r')]
testY = [array([float(x) for  x in line.split()[1:]]).reshape((d, 1)) for line in open('negationVectorsTest.txt', 'r')]

# # gradient descent
# def F(w):
#     return sum((x.dot(w) - y)**2 for x, y in points) / numExamples
# def dF(w):
#     return sum(2 * (x.dot(w) - y) * x for x, y in points) / numExamples


# stochastic gradient descent
def sF(w, i):
    x, y = trainX[i], trainY[i]
    #word = x[0]
    #x = array(x[1:])
    #wordNot = y[0]
    #y = array(y[1:])
    return (x.dot(Vnot.T)).dot(w)

def sdF(w, i):
    x, y = trainX[i], trainY[i]
    return x.dot(Vnot.T)

#gradientDescent(F, dF, d)
numExamples = len(trainX)
stochasticGradientDescent(sF, sdF, d, numExamples)
