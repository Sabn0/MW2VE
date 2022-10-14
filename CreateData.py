
# import packages
from collections import defaultdict, Counter
import numpy as np

def readLines(file_name: str):
    with open(file_name, 'r') as f:
        content = f.read().splitlines()
    content = [s.strip().lower() for s in content]
    return content


def getWords(sentences: list, vocab_size: int, min_freq: int, alpha_smooth: float) -> tuple:
    w2c = dict(Counter(' '.join(sentences).split()).most_common(vocab_size))
    w2c = {w: c**alpha_smooth for w, c in w2c.items() if c >= min_freq}
    total = sum(w2c.values())
    w2prob = {w: (c/total) for w, c in w2c.items()}
    w2i = {w: i for i, w in enumerate(list(w2c.keys()))}
    words = np.array(list(w2i.keys()))
    return w2i, words, w2prob


def sampleExamples(sentences: list, w2prob: dict, window: int, K_down_sample: int, max_sen_length=30) -> list:

    assert window > 0 , "invalid window size {}".format(window)
    vocab_words, probs = list(w2prob.keys()), list(w2prob.values())

    triplets = []

    for j, sentence in enumerate(sentences):
        words = sentence.strip().split()
        if len(words) > max_sen_length: continue
        for i in range(window, len(words)-window):

            # down-sampling: skip frequent words half of the time
            word = words[i]
            if word not in vocab_words or (w2prob[word] >= K_down_sample and np.random.choice(a=[0,1], p=[0.5, 0.5])):
                continue

            # sample contexts
            rng = list(range(i-window, i)) + list(range(i+1, i+window+1))
            context = words[np.random.choice(rng)]
            rnd_word = np.random.choice(a=vocab_words, p=probs)
            if any(w not in vocab_words for w in [context, rnd_word]):
                continue

            triplets += [(word, context, rnd_word)]

    return triplets

