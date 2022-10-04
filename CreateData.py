
# import packages
from collections import defaultdict, Counter
import numpy as np


def readLines(file_name: str):
    with open(file_name, 'r') as f:
        content = f.read().splitlines()
    return content

def getWords(sentences: list, vocab_size: int) -> tuple:
    words = list(dict(Counter(' '.join(sentences).split()).most_common(vocab_size)).keys())
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

def makeExamples(sentences: list, window: int):
    assert window > 0 , "invalid window size {}".format(window)
    for sentence in sentences:
        words = sentence.strip().split()
        for i in range(window, len(words)-window):
            rng = list(range(i-window, i)) + list(range(i+1, i+window+1))
            near_word = np.random.choice(rng)
            yield words[i], words[near_word]