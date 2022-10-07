# MW2VE - English word2vec
A self learning experience with w2v. The program creates word vectors with skip-grams and negative sampling. \
Implenetation based on the original paper by [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781), and a following paper by [Goldberg and Levy, 2014](https://arxiv.org/abs/1402.3722).

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
When trained+tested on some wiki sentences for several epochs, with toy parameters (embedding=20, vocab size=1K, window=5):

| spoke     | number      |
| ------------- | ------------- |
| spoke          | number         |
| campaign          | two         |
| number          | whose         |
| sitting          | office         |
| Chamber          | group         |
| NDP          | many         |
| just          | rates         |
| seen          | area         |
| former          | spoke         |
| finance          | coming         |

<a href="url"><img src="https://user-images.githubusercontent.com/45892555/194540533-ae99a383-508d-4eda-9fba-e4e1d2471715.png" align="left" height="200" width="200" ></a>



