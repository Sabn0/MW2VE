# MW2VE
A self-learning experience with w2v. Creation of word vectors using skip-grams and negative sampling. \
Implenetation based on the [original paper](https://arxiv.org/abs/1301.3781), and [this following paper](https://arxiv.org/abs/1402.3722).

## Example
When trained for a few epochs on a very small sample of sentences from PennTreebank (loaded from torchtext) with small parameters (embedding=20, vocab size=500, window=5):

|     | |    |    |
| ----------| -------- | ------  | -------- |
| **credit** | cash | stock | contracts | technology |
| **street**    | state   |  off  |  government  | committee |
| **people**    | around   |  office  |  employees  | him |
