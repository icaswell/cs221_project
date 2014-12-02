from numpy import *
from numpy.random import *
from numpy.linalg import norm

############################################################
def stochasticGradientDescent(sF, sdF, d, n):
    w = zeros((d, 1))
    numIters = 1000
    eta = 0.01  # step size
    for t in range(numIters):
        print "t = %s, w = %s" % (t, w)
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

V_not = array([float(x) for i, x in enumerate(open('not.txt', 'r').readline().split()) if i])
d = V_not.shape[0]
V_not = V_not.reshape((d, 1))

def read_vocab_file(fname, d):
    return [(
        line.split()[0],
        array([float(x) for  x in line.split()[1:]]).reshape((d, 1))
    ) for line in open(fname, 'r')]

trainX = read_vocab_file('originalVectorsTrain.txt', d)
trainY = read_vocab_file('negationVectorsTrain.txt', d)
testX = read_vocab_file('originalVectorsTest.txt', d)
testY = read_vocab_file('negationVectorsTest.txt', d)

# stochastic gradient descent
# def sf(w, i):
#     x, y = trainX[i], trainY[i]
#     word, Vword = x
#     word_not, Vword_not = y
#     return (word_vec.dot(V_not.T)).dot(w) * counts[word_not]

# def sdf(w, i):
#     x, y = trainX[i], trainY[i]
#     word, Vword = x
#     word_not, Vword_not = y    
#     return Vword.dot(V_not.T)*counts[word_not]

def sF(w, x, y):
    word, Vword = x
    word_not, Vword_not = y
    f_word = (Vword.dot(V_not.T)).dot(w) #lol
    return -(1/counts[word_not]) * f_word.T.dot(Vword_not)/(norm(f_word) * norm(Vword_not))

def sdF(w, x, y):
    word, Vword = x
    word_not, Vword_not = y
    f_word = (Vword.dot(V_not.T)).dot(w)
    num_1 = norm(f_word)*norm(Vword_not) * Vword.T.dot(V_not).dot(Vword.T)
    num_2 = (norm(Vword_not)/norm(f_word)) * f_word.T.dot(Vword) * f_word
    denom = -counts[word_not] * (norm(f_word)*norm(Vword_not))**2
    return (num_1 - num_2)/denom
                                                        
    
#===============================================================================
# test this!
numExamples = len(trainX)
w_trained = stochasticGradientDescent(sF, sdF, d, numExamples)

w_rand = np.rand(w.shape)
w_uniform = np.ones(w.shape)
w_trained_inv = -w

w_candidates = [w, w_rand, w_uniform, w_trained_inv]
objective_scores = np.zeros((len(w_candidates), 1))

for i in range(len(testX)):
    _, x = testX[i]
    _, y = testY[i]
    objective_scores += array([sF(w, x, y) for w in w_candidates])

print objective_scores
    
