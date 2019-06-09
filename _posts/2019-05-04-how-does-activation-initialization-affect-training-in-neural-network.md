---
layout: post
title:  "How does activation initialization affect training in neural network"
date:   2019-05-04 19:24:42 +0530
tags: activation initialization
---


Initialization affect training in neural network

```python
import torch
def init():
    x = torch.randn(512)
    a = torch.randn(512, 512)
    return x,a

x,a = init()

for i in range(100): 
    x = a @ x

x.std(), x.mean() 

```




    (tensor(nan), tensor(nan))



The matrix calculation is simply blowing and pytorch is not able to keep up the calculation. it is giving us NaN.

Let's findout iteration count after which the x.std() is blown.



```python
x,a = init()
for i in range(100): 
    x = a @ x
    if torch.isnan(x.std()):
        break
i
```




    28



Lets keep activation low.


```python
x,a = init()
a *= 0.01
for i in range(100): 
    x = a @ x

x.std(), x.mean() 
```




    (tensor(0.), tensor(0.))



At the end of the multiplications we want to have std of 1 & mean of 0.


```python
size = 512
cnt = 100
def check():
    mean,sqr = 0.,0.
    for i in range(cnt):
        x = torch.randn(size)
        a = torch.randn(size, size)
        y = a @ x
        mean += y.mean().item()
        sqr  += y.pow(2).mean().item()
    print(mean/cnt,sqr/(cnt*size)) #mean & std


check()
size = 1
cnt = 1000
check()
size = 2
cnt = 1000
check()
```

    -0.013873589411377906 1.0052018302679062
    -0.048302864631841656 0.8950763777035272
    0.012876306110993028 0.9755765050308837


It looks like there is a link between the number of tensor shape and final mean, std we are calculating.
`x` & `a` both have a uniform distribution. And if we keep multiplying `x & a` then accumulated avg mean & avg std is close to 0 and 1

It looks like basis for [Kaiming initialization](https://arxiv.org/abs/1502.01852)

### Yes our initialization will affect our training of neural network.


```python

```
Thanks [fast.ai](https://www.fast.ai/) for this tutorial.