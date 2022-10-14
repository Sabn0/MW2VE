# MW2VE
A self-learning experience with w2v. Creation of word vectors using skip-grams and negative sampling. \
Implenetation based on the [original paper](https://arxiv.org/abs/1301.3781), and [this following paper](https://arxiv.org/abs/1402.3722).

## How to train
Run the Train.py script supplied with a sentences file (plain text with sentences).
```
python Train.py -s=PATH_TO_FILE
```
The program will train and create 3 output files of npy format - words, words vector and context vectors.\
Hyper-parameters can be changed manually.

## How to test
Run the Test.py script supplied with 3 npy files (output of the Train.py script) and a path to output file.
```
python Test.py -w=PATH_TO_WORDS -c=PATH_TO_WORD_VECTORS -c=PATH_TO_CONTEXT_VECTORS -o=PATH_TO_OUTPUT
```
The program will print the 10 most similar word to some hard-coded target words, and save a 2dim projection using PCA of the matrices sum.

## Example
When trained+tested on a sample of sentences from PennTreebank (loaded from torchtext) with toy parameters (embedding=20, vocab size=500, window=5):

|     | |    |    |   |
| ----------| -------- | ------  | -------- |
| **credit** | cash | stock | contracts | technology |
| **street**    | state   |  off  |  government  | committee |
| **people**    | around   |  office  |  employees  | him |
