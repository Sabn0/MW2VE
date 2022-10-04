
# import packages
import numpy as np
from typing import *
from Modules.SkipGramsModel import SkipGramsModel
from Modules.BaseW2VModel import BaseW2VModel
from CreateData import makeExamples, readLines, getWords
import argparse

def train(
        sentences: list,
        max_iter: int,
        learning_rate: float,
        embedding_dim: int,
        w2i: dict,
        window: int,
        model: BaseW2VModel
):

    # initialize random matrices
    mat_shape = (len(w2i), embedding_dim)
    _xavier = np.sqrt(6 / sum(mat_shape))
    E = np.random.uniform(-_xavier, _xavier, mat_shape)           # targets
    E_tag = np.random.uniform(-_xavier, _xavier, mat_shape)       # contexts

    for i in range(max_iter):
        print(i)
        count = loss = 0
        for (word, context) in makeExamples(sentences, window=window):

            if word not in w2i or context not in w2i:
                continue

            params = (E, E_tag, word, context)
            _loss, (dw, dc, dr) = model.computeLossAndGrads(params)
            loss += _loss
            count += 1

            E -= learning_rate * dw
            E_tag -= learning_rate * dc
            E_tag -= learning_rate * dr

        print(loss/count)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--Sentences', required=True, type=str, help='path to sentences file')
    args = parser.parse_args()

    window = 3
    max_iter = 100
    learning_rate = 0.01
    embedding_dim = 100
    max_vocab_size = int(1e03)

    sentences = readLines(args.Sentences)
    sentences = sentences[:1000]
    w2i, i2w = getWords(sentences, vocab_size=max_vocab_size)
    print("vocab size: {}".format(len(w2i)))
    model = BaseW2VModel(w2i)

    train(
        sentences=sentences,
        embedding_dim=embedding_dim,
        max_iter=max_iter,
        learning_rate=learning_rate,
        model=model,
        window=window,
        w2i=w2i
    )



if __name__ == "__main__":
    main()
