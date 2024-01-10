from math import sqrt
import random
import numpy as np

vocabulary_file='Exercise 3/word_embeddings.txt'

def dist(word1, word2):
    return sqrt(np.sum((word1-word2)**2))

# Find and return three closest words to any word
def findthreeclosest(word, W, word_idx, ivocab):
    word_vec = W[word_idx]
    distances = {}
    i = 0
    for vec in W:
        distances[i] = dist(word_vec, vec) # Index distance pair
        i += 1
    distances = sorted(distances.items(), key=lambda x:x[1]) # Sort by distance
    # Closest words
    closest = {}
    i = 0
    while i < 3:
        pair = distances[i]
        word = ivocab[pair[0]]
        closest[word] = pair[1]
        i += 1
    return closest

def main():
    # Read words
    print('Read words...')
    with open(vocabulary_file, 'r', encoding="utf8") as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]

    # Read word vectors
    print('Read word vectors...')
    with open(vocabulary_file, 'r', encoding="utf8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    # Vocabulary and inverse vocabulary (dict objects)
    print('Vocabulary size')
    print(len(vocab))
    print(vocab['man'])
    print(len(ivocab))
    print(ivocab[10])

    # W contains vectors for
    print('Vocabulary word vectors')
    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v
    print(W.shape)
        
    # Main loop for analogy
    while True:
        input_term = input("\nEnter three words (EXIT to break): ")
        if input_term.lower() == 'exit':
            break
        else:
            input_words = input_term.split(" ")
            for word in input_words:
                idx = vocab[word]
                threeclosest = findthreeclosest(word, W, idx, ivocab)
                print("Three closest words of word", word, ":")
                print("\n                               Word       Distance\n")
                print("---------------------------------------------------------\n")
                for word, distance in threeclosest.items():
                    print("%35s\t\t%f\n" % (word, distance))

main()


