import builtins
import torch
import torchtext
import collections
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = None
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Loads AG News dataset
# and initializes the vocabulary properly.
def load_dataset(ngrams=1, min_freq=1):
    global vocab, tokenizer
    print("Loading dataset...")
    train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    print('Building vocab...')
    counter = collections.Counter()
    for (label, line) in train_dataset:
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line), ngrams=ngrams))
    vocab = torchtext.vocab.vocab(counter, min_freq=min_freq)
    return train_dataset, test_dataset, classes, vocab

stoi_hash = {}

# Encodes textual data
# using tokenizer and vocabulary.
def encode(x, voc=None, unk=0, tokenizer=tokenizer):
    global stoi_hash
    v = vocab if voc is None else voc
    if v in stoi_hash.keys():
        stoi = stoi_hash[v]
    else:
        stoi = v.get_stoi()
        stoi_hash[v] = stoi
    return [stoi.get(s, unk) for s in tokenizer(x)]

# Trains a single epoch
# for the provided neural network.
def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for labels, features in dataloader:
        optimizer.zero_grad()
        features, labels = features.to(device), labels.to(device)
        out = net(features)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
        i += 1
        if i % report_freq == 0:
            print(f"{count}: acc={acc.item() / count}")
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count

# Pads sequences
# up to maximum length.
def padify(b, voc=None, tokenizer=tokenizer):
    v = [encode(x[1], voc=voc, tokenizer=tokenizer) for x in b]
    l = max(map(len, v))
    return (
        torch.LongTensor([t[0] - 1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t), (0, l - len(t)), mode='constant', value=0) for t in v])
    )

# Computes offsets
# for unpadded sequences batch.
def offsetify(b, voc=None):
    x = [torch.tensor(encode(t[1], voc=voc)) for t in b]
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    return (
        torch.LongTensor([t[0] - 1 for t in b]),
        torch.cat(x),
        o
    )

# Trains single embedding epoch
# using offsets instead of padding.
def train_epoch_emb(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None, report_freq=200, use_pack_sequence=False):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for labels, text, off in dataloader:
        optimizer.zero_grad()
        labels, text = labels.to(device), text.to(device)
        if use_pack_sequence:
            off = off.to('cpu')
        else:
            off = off.to(device)
        out = net(text, off)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
        i += 1
        if i % report_freq == 0:
            print(f"{count}: acc={acc.item() / count}")
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count
