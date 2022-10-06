
# import packages
import numpy as np
import argparse
from scipy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler

def normalizeRows(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1)
    W /= norms[:, np.newaxis]
    #W = normalize(MinMaxScaler().fit_transform(W), axis=1, norm='l1')
    return W

def getSimilarWords(words: np.ndarray, vectors: np.ndarray, target_vector: np.ndarray, K: int) -> list:

    # normalize if vectors are not normalized
    epsilon = float('3e-03')
    if not all(np.linalg.norm(e) - epsilon < 1 < np.linalg.norm(e) + epsilon for e in vectors):
        vectors = normalizeRows(vectors)

    # get most similar K words to target
    similarities = np.dot(vectors, target_vector)
    sorted_sims = similarities.argsort()[::-1]
    best_sims = [(words[sorted_sims[i]], similarities[sorted_sims[i]]) for i in range(K)]
    return best_sims


def plot2Dim(E: np.ndarray, words: np.ndarray, N_words_use: int) -> None:

    """Implementation from https://stats.stackexchange.com/questions/235882/pca-in-numpy-and-sklearn-produces-different-results"""
    random_indexes = np.random.choice(len(words), N_words_use)
    E, words = E[random_indexes], words[random_indexes]
    E -= np.mean(E, axis=0)

    # calculate eigenvalues and eigenvectors
    cov = (1. / len(E)) * E.T.dot(E)
    evals, evecs = LA.eigh(cov)
    # sort them in descending order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    # Project word vectors onto the first 2 PCs, and then visualize. Use matplotlib.pyplot.scatter
    projection = np.dot(E, evecs[:, :2])
    plt.figure(figsize=(13, 7))
    plt.scatter(projection[:, 0], projection[:, 1])
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)
    """    plt.ylim(-1, 1)
        plt.xlim(-1, 1)"""
    plt.title("word vectores projected to first 2 PCs")
    for i, word in enumerate(words):
        plt.annotate(word, xy=(projection[i, 0], projection[i, 1]))
    plt.show()


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-w', '--WordsMatrix', required=True, type=str, help='path to npy matrix')
    parser.add_argument('-v', '--VectorsMatrix', required=True, type=str, help='path to npy matrix')
    parser.add_argument('-c', '--ContextsMatrix', required=True, type=str, help='path to npy matrix')
    args = parser.parse_args()

    # hyper params
    targets = ['job']
    K = 10
    N_words_plot = 50

    words = np.load(args.WordsMatrix)
    word_vectors = np.load(args.VectorsMatrix)
    context_vectors = np.load(args.ContextsMatrix)
    # normalize vectors
    word_vectors = normalizeRows(word_vectors)
    context_vectors = normalizeRows(context_vectors)

    # print K most common words to targets set
    for target in targets:
        target_vector = word_vectors[np.where(words == target)[0][0]]
        sims = getSimilarWords(words=words, vectors=context_vectors, target_vector=target_vector, K=K)
        for (sim_word, score) in sims:
            print("similar word: {}, score: {}".format(sim_word, score))

    # plot PCA 2d for words
    E = word_vectors + context_vectors
    plot2Dim(E, words=words, N_words_use=N_words_plot)


if __name__ == "__main__":
    main()







