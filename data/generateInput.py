import os
from collections import Counter

mapping = dict()

mapping[(3, 2)] = 1
mapping[(4, 1)] = 1
mapping[(5, 0)] = 1
mapping[(1, 4)] = 0
mapping[(0, 5)] = 0

out = open('input.txt', 'w')

with open('dev.data') as f:
    for line in f.readlines():
        parts = line.split('\t')
        sent1 = parts[2]
        sent2 = parts[3]
        givenLable = (int(parts[4][1]), int(parts[4][4]))
        if givenLable == (2,3):
            continue
        else:
            label = mapping[givenLable]
        out.write(str(label)+"\n")
        out.write(sent1+"\n")
        out.write(sent2+"\n")


out.close()
    


