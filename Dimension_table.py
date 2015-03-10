__author__ = 'Jeff'

f = open("/Users/Jeff/Desktop/DataHack/word2vec.txt", 'r')

master_dictionary = {}
input_vector_words = []
input_vector_scores = []
index_count = 0
overlap = 0

for i,line in enumerate(f.readlines()):
    input_vector_word = []
    input_vector_score = []
    words = line.split(", ")
    
    for j, word in enumerate(words):
        values = word.split(" ")
        
        try:
            this_word = values[0]
            this_score = float(values[1])
            input_vector_word.append(this_word)
            input_vector_score.append(this_score)
            
            if this_score > 0.003:   #<--- fine tune it to get proper dimentions
                if this_word not in master_dictionary:
                    master_dictionary[this_word] = index_count
                    index_count += 1
                else:
                    overlap += 1
                    
        except:
            pass

    input_vector_words.append(input_vector_word)
    input_vector_scores.append(input_vector_score)

print len(master_dictionary)
print "overlap: " + str(overlap)

sparse_X = []

master_set = set(master_dictionary.keys())

for i,document_word_vector in enumerate(input_vector_words):
    each_X = [0] * len(master_set)
    document_word_set = set(document_word_vector)
    difference_set = master_set - document_word_set
    difference_list = list(difference_set)
    
    for word in master_dictionary.keys():
        master_word_index = master_dictionary[word]
        each_X[master_word_index] = word
        
        if word in document_word_set:
            this_doc_word_index = input_vector_words[i].index(word)
            each_X[master_word_index] = input_vector_scores[i][this_doc_word_index]
        
        else:
            each_X[master_word_index] = 0.00
    
    sparse_X.append(each_X)


text_file = open("/Users/Jeff/Desktop/DataHack/input.txt", "w")
for n in range(len(sparse_X)):
    for d in range(len(sparse_X[0])):
        text_file.write(str(sparse_X[n][d]) + "\t")
    text_file.write("\n")
text_file.close()


for n in range(len(sparse_X)):
        print sparse_X[n]



f.close()


#END
