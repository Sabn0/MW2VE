
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
        uniform_choice=0.5
) -> tuple:

    # initialize random matrices
    mat_shape = (len(w2i), embedding_dim)
    E = np.random.uniform(-uniform_choice, uniform_choice, mat_shape) / embedding_dim           # targets
    E_tag = np.random.uniform(-uniform_choice, uniform_choice, mat_shape) / embedding_dim       # contexts

    # history
    loss_train = []
    lr_changed = False
    this_time = time.time()

    for i in range(max_iter):
        print("epoch {}".format(i))
        count = loss = 0
        for (sen_id, word, context, rnd_word) in sampleExamples(sentences, w2prob=w2prob, window=window, K_down_sample=k_down):

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
    max_iter = 100
    learning_rate = 0.1
    embedding_dim = 20
    max_vocab_size = int(1e03)
    alpha_smooth = 0.75
    min_freq = 10
    N_sentences = int(2e04)

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
