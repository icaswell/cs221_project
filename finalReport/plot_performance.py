# graph_data.py

import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '{0:.2f}'.format(height),
                ha='center', va='bottom')

#========================================
if 0:
    #Enniliye Maha Oliyo
    fig, ax = plt.subplots()
    width = 0.15
    domain = np.arange(4)

    Precision = [0.665, 0.594, 0.643, 0.645]
    Recall = [0.422, 0.388, 0.4, 0.4]
    F1   = [0.517, 0.47, 0.493, 0.494]
    Accuracy = [0.754, 0.727, 0.744, 0.745]

    prec_bar = ax.bar(domain, Precision, width, color='g')
    rec_bar = ax.bar(domain + 1*width, Recall, width, color='k')
    f1_bar = ax.bar(domain + 2*width, F1, width, color='b')
    acc_bar = ax.bar(domain + 3*width, Accuracy, width, color='r')

    ax.legend((prec_bar[0], rec_bar[0], f1_bar[0], acc_bar[0]), \
        ('Precision', 'Recall', 'F1',  'Accuracy'), loc='best',  prop={'size':10})

    ax.set_xticks(domain + 2*width) 
    ax.set_xticklabels(('wiki_giga', 'googleNews', 'twitterGlove', 'wiki_giga_skipgram'))

    plt.ylim([0,1])

    plt.title("Performance by type of word vectors")
    #plt.ylabel('Pct correctly guessed for 50 predictions')
    autolabel(prec_bar)
    autolabel(rec_bar)
    autolabel(f1_bar)
    autolabel(acc_bar)
    plt.show()


if 1:

    fig, ax = plt.subplots()
    width = 0.15

    domain = np.arange(4)
    Precision = [0.665, 0.667, 0.67, 0.656]
    Recall = [0.422, 0.405, 0.412, 0.217]
    F1   = [0.517, 0.504, 0.511, 0.326]
    Accuracy =[0.754, 0.752, 0.754, 0.74]

    prec_bar = ax.bar(domain, Precision, width, color='g')
    rec_bar = ax.bar(domain + 1*width, Recall, width, color='k')
    f1_bar = ax.bar(domain + 2*width, F1, width, color='b')
    acc_bar = ax.bar(domain + 3*width, Accuracy, width, color='r')

    #ax.legend((bar_AA[0], bar_SRW[0], bar_MN[0], bar_PPR[0]), ('Adamic Adar', 'SRW', 'Mutual Neighbors', 'PPR') )
    ax.legend((prec_bar[0], rec_bar[0], f1_bar[0], acc_bar[0]), ('Precision', 'Recall', 'F1',  'Accuracy'), loc = 'best', prop={'size':10})

    ax.set_xticks(domain + 2*width) 
    ax.set_xticklabels(('Logistic Regression', 'LinearSVM', 'RBF SVM', 'RAE'))

    plt.ylim([0,1])

    plt.title("Performace by predictor")
    #plt.ylabel('Pct correctly guessed for 50 predictions')
    autolabel(prec_bar)
    autolabel(rec_bar)
    autolabel(f1_bar)
    autolabel(acc_bar)

    plt.show()


if 0:

    fig, ax = plt.subplots()
    width = 0.15

    domain = np.arange(4)
    Precision = [0.102, 0.576, 0.576, 0.665]
    Recall = [0.134, 0.398, 0.406, 0.422]    
    F1   = [0.115, 0.471, 0.476, 0.517]
    Accuracy =[0.689, 0.722, 0.722, 0.754]

    prec_bar = ax.bar(domain, Precision, width, color='g')
    rec_bar = ax.bar(domain + 1*width, Recall, width, color='k')
    f1_bar = ax.bar(domain + 2*width, F1, width, color='b')
    acc_bar = ax.bar(domain + 3*width, Accuracy, width, color='r')

    #ax.legend((bar_AA[0], bar_SRW[0], bar_MN[0], bar_PPR[0]), ('Adamic Adar', 'SRW', 'Mutual Neighbors', 'PPR') )
    ax.legend((prec_bar[0], rec_bar[0], f1_bar[0], acc_bar[0]), ('Precision', 'Recall', 'F1',  'Accuracy'), loc = 'best', prop={'size':10})

    ax.set_xticks(domain + 2*width) 
    ax.set_xticklabels(('l2sim','+ cosine sim','+ NER','+ ngram'))

    plt.ylim([0,1])

    plt.title("Feature Selection")
    #plt.ylabel('Pct correctly guessed for 50 predictions')
    autolabel(prec_bar)
    autolabel(rec_bar)
    autolabel(f1_bar)
    autolabel(acc_bar)

    plt.show()




