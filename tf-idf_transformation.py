__author__ = 'Jeff'


import math
from textblob import TextBlob as tb


def tf(word, blob):
    return blob.words.count(word) / float(len(blob.words))

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / float((1 + n_containing(word, bloblist))))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


p = open("/Users/Jeff/Desktop/DataHack/word2vec.txt", 'w')

with open ("/Users/Jeff/Desktop/DataHack/page.txt", "r") as myfile:
    data=myfile.readlines()


bloblist = []
for line in data:
    blo = tb(line.decode('utf-8'))
    bloblist.append(blo)

for i, blob in enumerate(bloblist):
    print""
    print i
    print""

    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for word in sorted_words:
        print "{} {}".format(word[0].encode('utf-8'),word[1])
        p.write(word[0].encode('utf-8') + " " + str(word[1]) + ", ")
    p.write("\n")

p.close()


#END
