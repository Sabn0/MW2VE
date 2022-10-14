

if __name__ == "__main__":

    import re
    import torch
    import torch.utils.data as data
    from torchtext.legacy.data import Field
    from torchtext.legacy.datasets import WikiText2

    text = Field(lower=True, batch_first=True, sequential=True, include_lengths=True)
    train, __, _ = WikiText2.splits(text_field=text)

    text = ' '.join(train.examples[0].text)
    text = re.sub(r'[^\x00-\x7F]+', '<non-ascii>', text)

    sentences = [s.strip() for s in text.split('<eos>') if len(s.strip())]

    with open('.data/sentences.txt', 'w+') as f:
        for sentence in sentences:
            f.write("%s\n" % sentence)

