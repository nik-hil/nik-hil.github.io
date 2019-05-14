---
layout: post
title:  "How to train a model in Pytorch or fastai from scratch"
date:   2019-05-06 12:24:42 +0530
tags: [Pytorch, fastai, training]
---

What are the steps to train my model in neural network? It is not very different than what we have 
learnt in our school days. Question was,

 1. Given a Simple quadratic function find the minima.
 1. We will draw the function.
 1. Choose a point and find the slope at that point.
 1. Move in direction where minima is less.
 1. We repeat above steps till we find a point were slope was zero.

 Ofcourse this was in school and problems were easy. In neural network we do the same.
 
```python
# Only required in google colab 
!curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    Done.

```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```

# Lets Import code 

```python
import operator

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')

from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)
    
def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train, y_train, x_valid, y_valid))

def normalize(x, mean, std_dev):
    return (x-mean)/std_dev
```


```python
from torch import nn
import torch.nn.functional as F
mpl.rcParams['image.cmap'] = 'gray'
```

We get our training and validation data
```python
x_train,y_train,x_valid,y_valid = get_data()
```


```python
n, m = x_train.shape
c = y_train.max() + 1
nh = 50
```

# Lets create our Model

```python
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in, nh), nn.ReLU(), nn.Linear(nh, n_out)]
        
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
```


```python
model = Model(m, nh, 10)
pred = model(x_train)
```


```python
def log_softmax(x):
    exp = x.exp()
    return (exp/exp.sum(-1, keepdim=True)).log()
```


```python
sm_pred = log_softmax(pred)
```


```python
def nll(inp, targ): # -ve log likelihood
    return -inp[range(targ.shape[0]), targ].mean()
```
# Higher dimension indexing.

Here we are using higher dimension indexing.
Numpy support this.

https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#integer-array-indexing

At times it can be very confusing.

```python
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> x[[0, 1, 2], [0, 1, 0]]
array([1, 4, 5])
```
 
# Lets check our loss

```python
loss = nll(sm_pred, y_train)
loss
```




    tensor(2.3060, grad_fn=<NegBackward>)



log_softmax is similified with [log sum exp trick](https://en.wikipedia.org/wiki/LogSumExp)




```python
def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()
```


```python
test_near(logsumexp(pred), pred.logsumexp(-1))
```

Same is availabel in Pytorch as F.nll_loss, F.log_softmax.

Using both F.cross_entropy is built


```python
test_near(F.cross_entropy(pred, y_train), loss)
```

# Training Loop

A training loop will do the following


1.   init all param in model
1.   Calculate y_pred from input & model
2.   calculate loss
3.   Claculate the gradient wrt to every param in model
4.   update those param 
4.   Repeat




```python
loss_func = F.cross_entropy

def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()
```


```python
accuracy(pred, y_train), accuracy(pred[:10], y_train[:10])
```




    (tensor(0.0981), tensor(0.))



Lets create a training loop


```python
lr = .5
epochs = 1
bs = 64
n
```




    50000


# We create our training loop.

```python
for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        # grab a batch of x & y from start_i to end_i
        xb = x_train[start_i: end_i]
        yb = y_train[start_i: end_i]
        # predict output and calculate loss
        loss = loss_func(model(xb), yb)
        
        loss.backward()
        with torch.no_grad():
            for l in model.layers:
                if hasattr(l, 'weight'):
                    l.weight -= l.weight.grad * lr
                    l.bias   -= l.bias.grad   * lr
                    l.weight.grad.zero_()
                    l.bias.grad.zero_()

```


```python
loss_func(model(xb), yb), accuracy(model(xb), yb)
```




    (tensor(0.1773, grad_fn=<NllLossBackward>), tensor(0.9375))




```console
nn.ModuleList??
```


```

```
