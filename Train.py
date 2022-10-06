
# import packages
import numpy as np
from Modules.NS_SkipGrams_w2vModel import w2vModel
from CreateData import sampleExamples, readLines, getWords
import argparse

def train(
        sentences: list,
        max_iter: int,
        learning_rate: float,
        embedding_dim: int,
        w2i: dict,
        w2prob: dict,
        window: int,
        model: w2vModel
) -> tuple:

    # initialize random matrices
    mat_shape = (len(w2i), embedding_dim)
    E = np.random.uniform(-0.5, 0.5, mat_shape) / embedding_dim           # targets
    E_tag = np.random.uniform(-0.5, 0.5, mat_shape) / embedding_dim       # contexts

    for i in range(max_iter):
        print(i)
        count = loss = 0
        for (word, context, rnd_word) in sampleExamples(sentences, w2prob=w2prob, window=window):

            params = (E, E_tag, word, context, rnd_word)
            _loss, (dw, dc, dr) = model.computeLossAndGrads(params)
            loss += _loss
            count += 1

            E -= learning_rate * dw
            E_tag -= learning_rate * dc
            E_tag -= learning_rate * dr

        a = np.linalg.norm(E[w2i['young']])
        b = np.linalg.norm(E[w2i['children']])
        c = np.dot(E[w2i['young']],E[w2i['children']])
        print("similairty between young and children: {}".format(c/(a*b)))
        print("epoch loss: {}".format(loss/count))

    return E, E_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--Sentences', required=True, type=str, help='path to sentences file')
    args = parser.parse_args()

    window = 3
    max_iter = 10
    learning_rate = 0.01
    embedding_dim = 50
    max_vocab_size = int(1e04)
    alpha_smooth = 0.75
    min_freq = 10

    sentences = readLines(args.Sentences)
    sentences = sentences[:2000]
    w2i, words, w2prob = getWords(sentences, vocab_size=max_vocab_size, min_freq=min_freq, alpha_smooth=alpha_smooth)
    print("vocab size: {}".format(len(w2i)))
    model = w2vModel(w2i)

    E, E_tag = train(
        sentences=sentences,
        embedding_dim=embedding_dim,
        max_iter=max_iter,
        learning_rate=learning_rate,
        model=model,
        window=window,
        w2i=w2i,
        w2prob=w2prob
    )

    # save matrices
    np.save('words.npy', words)
    np.save('words_v.npy', E)
    np.save('contexts_v.npy', E_tag)


if __name__ == "__main__":
    main()
