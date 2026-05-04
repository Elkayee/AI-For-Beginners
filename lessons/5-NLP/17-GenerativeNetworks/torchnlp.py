import builtins
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torchtext.datasets import AG_NEWS
import collections
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_obj = None
tokenizer = get_tokenizer('basic_english')
stoi_hash = {}

def get_ag_news():
    # Load AG NEWS dataset
    # Returns dataset lists and classes
    train_ds, test_ds = AG_NEWS(root='./data')
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    return list(train_ds), list(test_ds), classes

def build_vocabulary(dataset, ngrams, min_freq):
    # Build vocab from dataset
    # Returns constructed vocabulary object
    from torchtext.data.utils import ngrams_iterator
    counter = collections.Counter()
    for (label, line) in dataset:
        counter.update(ngrams_iterator(tokenizer(line),ngrams=ngrams))
    return vocab(counter, min_freq=min_freq)

def load_dataset(ngrams=1,min_freq=1):
    # Load dataset and build vocabulary
    # Sets global vocab variable
    global vocab_obj, tokenizer
    train_dataset, test_dataset, classes = get_ag_news()
    vocab_obj = build_vocabulary(train_dataset, ngrams, min_freq)
    return train_dataset, test_dataset, classes, vocab_obj

def get_stoi(v):
    # Get string to index mapping
    # Caches mapping for performance
    global stoi_hash
    if v not in stoi_hash:
        stoi_hash[v] = v.get_stoi()
    return stoi_hash[v]

def encode(x,voc=None,unk=0,tokenizer=tokenizer):
    # Encode text sequence into integers
    # Uses tokenizer and vocabulary mapping
    v = vocab_obj if voc is None else voc
    stoi = get_stoi(v)
    return [stoi.get(s,unk) for s in tokenizer(x)]

def process_batch_nlp(net, features, labels, loss_fn, optimizer=None):
    # Process a single batch
    # Compute loss and accuracy
    out = net(features)
    loss = loss_fn(out,labels)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _,predicted = torch.max(out,1)
    acc = (predicted==labels).sum()
    return loss, acc, len(labels)

def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.CrossEntropyLoss(),epoch_size=None, report_freq=200):
    # Train model for one epoch
    # Returns average loss and accuracy
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels,features in dataloader:
        features, labels = features.to(device), labels.to(device)
        loss, a, c = process_batch_nlp(net, features, labels, loss_fn, optimizer)
        total_loss+=loss
        acc+=a
        count+=c
        i+=1
        if i%report_freq==0:
            print(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return total_loss.item()/count, acc.item()/count

def padify(b,voc=None,tokenizer=tokenizer):
    # Pad sequences to max length
    # Returns labels and features tensors
    v = [encode(x[1],voc=voc,tokenizer=tokenizer) for x in b]
    l = max(map(len,v))
    lbls = torch.LongTensor([t[0]-1 for t in b])
    feats = torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=0) for t in v])
    return lbls, feats

def offsetify(b,voc=None):
    # Create offsets for embedding bag
    # Returns labels texts and offsets
    x = [torch.tensor(encode(t[1],voc=voc)) for t in b]
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    lbls = torch.LongTensor([t[0]-1 for t in b])
    return lbls, torch.cat(x), o

def process_batch_emb(net, text, off, labels, loss_fn, optimizer=None):
    # Process batch with embeddings
    # Compute loss and accuracy
    out = net(text, off)
    loss = loss_fn(out,labels)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _,predicted = torch.max(out,1)
    acc = (predicted==labels).sum()
    return loss, acc, len(labels)

def train_epoch_emb(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.CrossEntropyLoss(),epoch_size=None, report_freq=200,use_pack_sequence=False):
    # Train epoch using embedding layer
    # Returns average loss and accuracy
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels,text,off in dataloader:
        labels,text = labels.to(device), text.to(device)
        off = off.to('cpu') if use_pack_sequence else off.to(device)
        loss, a, c = process_batch_emb(net, text, off, labels, loss_fn, optimizer)
        total_loss+=loss
        acc+=a
        count+=c
        i+=1
        if i%report_freq==0:
            print(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return total_loss.item()/count, acc.item()/count
