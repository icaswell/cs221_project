"""
@Author: Isaac Caswell, Melvin J.J.J. Premkumar, Percy Liang
@Created: 30 Nov.

Performs SGD on our function and then tests the resultant w.  More doc to come, one hopes.
"""

from numpy import *
from numpy.random import *
from numpy.linalg import norm

############################################################
## 
def stochasticGradientDescent(sF, sdF, d, n, numIters = 1000):
    #w = zeros((d, 1))
    w = random((d, 1))
    eta = 0.01  # step size
    for t in range(numIters):
        print 'gradient descent iteration {0} of {1}...'.format(t + 1, numIters)
        #print "t = %s, w = %s" % (t, w)
        for i in range(n):
            x, y = trainX[i], trainY[i]
            value = sF(w, x, y)
            gradient = sdF(w, x, y)
            w = w - eta * gradient
    return w

############################################################
## create map from word to count
counts = {}
with open('vocab.txt', 'r') as countF:
    for line in countF:
        line = line.split()
        counts[line[0]] = int(line[1])

############################################################
## read in the training and test sets and the negation vector

Vnot = array([float(x) for i, x in enumerate(open('not.txt', 'r').readline().split()) if i])
d = Vnot.shape[0]
Vnot = Vnot.reshape((d, 1))

def read_vocab_file(fname, d):
    return [(
        line.split()[0],
        array([float(x) for  x in line.split()[1:]]).reshape((d, 1))
    ) for line in open(fname, 'r')]

trainX = read_vocab_file('originalVectorsTrain.txt', d)
trainY = read_vocab_file('negationVectorsTrain.txt', d)
testX = read_vocab_file('originalVectorsTest.txt', d)
testY = read_vocab_file('negationVectorsTest.txt', d)

############################################################
## define the objective function and the derivative thereof

# stochastic gradient descent
# def sf(w, i):
#     x, y = trainX[i], trainY[i]
#     word, Vword = x
#     word_not, Vword_not = y
#     return (word_vec.dot(Vnot.T)).dot(w) * counts[word_not]

# def sdf(w, i):
#     x, y = trainX[i], trainY[i]
#     word, Vword = x
#     word_not, Vword_not = y    
#     return Vword.dot(Vnot.T)*counts[word_not]

def sF(w, x, y):
    word, Vword = x
    word_not, Vword_not = y
    f_word = Vword.dot(Vnot.T).dot(w) # lol
    return -(1/counts[word_not]) * f_word.T.dot(Vword_not)/(norm(f_word) * norm(Vword_not))

def sdF(w, x, y):
    word, Vword = x
    word_not, Vword_not = y
    f_word = (Vword.dot(Vnot.T)).dot(w) # dx1
    num_1 = norm(f_word)*norm(Vword_not) * Vword.dot(Vnot.T).dot(Vword) # dx1
    num_2 = (norm(Vword_not)/norm(f_word)) * f_word.T.dot(Vword) * f_word # dx1
    denom = -counts[word_not] * (norm(f_word)*norm(Vword_not))**2
    return (num_1 - num_2)/denom
############################################################
## Test it versus random w!

numExamples = len(trainX)
w_trained = stochasticGradientDescent(sF, sdF, d, numExamples, 1)

w_rand = random(w_trained.shape)
w_uniform = ones(w_trained.shape)
w_trained_inv = -w_trained

w_candidates = [w_trained, w_rand, w_uniform, w_trained_inv]
print [i.shape for i in w_candidates]
objective_scores = zeros((len(w_candidates), 1))

for i in range(len(testX)):
    x = testX[i]
    y = testY[i]
    objective_scores += array([sF(w, x, y) for w in w_candidates])

print objective_scores
    
