
# import packages
from collections import defaultdict, Counter
import numpy as np


def readLines(file_name: str):
    with open(file_name, 'r') as f:
        content = f.read().splitlines()
    return content

def getWords(sentences: list, vocab_size: int, min_freq: int, alpha_smooth: float) -> tuple:
    w2c = dict(Counter(' '.join(sentences).split()).most_common(vocab_size))
    w2c = {w: c**alpha_smooth for w, c in w2c.items() if c >= min_freq}
    total = sum(w2c.values())
    w2prob = {w: (c/total) for w, c in w2c.items()}
    w2i = {w: i for i, w in enumerate(list(w2c.keys()))}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w, w2prob

def sampleExamples(sentences: list, w2prob: dict, window: int):
    assert window > 0 , "invalid window size {}".format(window)
    vocab_words, probs = list(w2prob.keys()), list(w2prob.values())
    for sentence in sentences:
        words = sentence.strip().split()
        for i in range(window, len(words)-window):
            rng = list(range(i-window, i)) + list(range(i+1, i+window+1))
            context = words[np.random.choice(rng)]
            rnd_word = np.random.choice(a=vocab_words, p=probs)
            word = words[i]
            if any(w not in vocab_words for w in [word, context, rnd_word]):
                continue
            yield word, context, rnd_word