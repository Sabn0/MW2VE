
# import packages
import numpy as np
from Modules.NS_SkipGrams_w2vModel import w2vModel
from CreateData import sampleExamples, readLines, getWords
import argparse
import time

def train(
        sentences: list,
        max_iter: int,
        learning_rate: float,
        embedding_dim: int,
        w2i: dict,
        w2prob: dict,
        window: int,
        model: w2vModel,
        k_down: int,
        uniform_choice=0.05
) -> tuple:

    # initialize random matrices
    mat_shape = (len(w2i), embedding_dim)
    x = np.sqrt(6/(np.sum(mat_shape)))
    E = np.random.uniform(-x,x, mat_shape)
    E_tag = np.random.uniform(-x, x, mat_shape)

    # history
    loss_train = []
    lr_changed = False
    this_time = time.time()

    # draw examples
    triples = sampleExamples(sentences, w2prob=w2prob, window=window, K_down_sample=k_down)

    for i in range(max_iter):
        print("epoch {}".format(i))
        count = loss = 0
        for (word, context, rnd_word) in triples:
            params = (E, E_tag, word, context, rnd_word)
            _loss, (dw, dc, dr) = model.computeLossAndGrads(params)
            loss += _loss
            count += 1

            E -= learning_rate * dw
            E_tag -= learning_rate * dc
            E_tag -= learning_rate * dr

        print("epoch loss: {}, time: {}".format(loss / count, time.time()-this_time))
        this_time = time.time()
        loss_train += [loss / count]

        # to next iteration
        if i == 0 or loss_train[-2] > loss_train[-1]:
            continue

        # break
        if lr_changed:
            break

        # update lr
        learning_rate /= 10
        lr_changed = True
        print("changed lr")

    return E, E_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--Sentences', required=True, type=str, help='path to sentences file')
    args = parser.parse_args()

    window = 5
    max_iter = 50
    learning_rate = 0.1
    embedding_dim = 20
    max_vocab_size = int(5e02)
    alpha_smooth = 0.75
    min_freq = 5
    N_sentences = int(1e04)

    sentences = readLines(args.Sentences)[:N_sentences]
    w2i, words, w2prob = getWords(sentences, vocab_size=max_vocab_size, min_freq=min_freq, alpha_smooth=alpha_smooth)
    print("vocab size: {}".format(len(w2i)))
    model = w2vModel(w2i)
    K_down_sample = int(len(w2i) * 0.1)

    E, E_tag = train(
        sentences=sentences,
        embedding_dim=embedding_dim,
        max_iter=max_iter,
        learning_rate=learning_rate,
        model=model,
        window=window,
        w2i=w2i,
        w2prob=w2prob,
        k_down=K_down_sample
    )

    # save matrices
    dir_name = 'Results'
    import os
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    np.save(os.path.join(dir_name, 'words.npy'), words)
    np.save(os.path.join(dir_name, 'words_v.npy'), E)
    np.save(os.path.join(dir_name, 'contexts_v.npy'), E_tag)


if __name__ == "__main__":
    main()
