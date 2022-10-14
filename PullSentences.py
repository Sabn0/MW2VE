

if __name__ == "__main__":

    import re
    import torch
    import torch.utils.data as data
    from torchtext.legacy.data import Field
    from torchtext.legacy.datasets import PennTreebank

    text = Field(lower=True, batch_first=True, sequential=True, include_lengths=True)
    train, __, _ = PennTreebank.splits(text_field=text)

    text = ' '.join(train.examples[0].text)
    text = re.sub(r'[^\x00-\x7F]+', '<non-ascii>', text)

    sentences = [s.strip() for s in text.split('<eos>') if len(s.strip())]
    sentences = [s for s in sentences if s.strip().split()[0].isalpha()]

    with open('.data/sentences.txt', 'w+') as f:
        for sentence in sentences:
            f.write("%s\n" % sentence)

