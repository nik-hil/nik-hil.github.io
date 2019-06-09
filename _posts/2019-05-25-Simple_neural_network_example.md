---
layout: post
title:  "Simplest neural network example in Fastai/Pytorch"
date:   2019-05-25 12:24:42 +0530
tags: visualization  pytorch fastai training example
---


What can be a simplest neural network example?


```python
# only on google colab
!curl -s https://course.fast.ai/setup/colab | bash
```
```console
    Updating fastai...
    Done.
```

We will implement simplest neural network with simple example of a line.

A line is represented as 


$ y = ax + b $

$ y = a_1x + a_2 x$

$ y = a_1x_1 + a_2 x_2$

$ y_i = a_1x_{1i} + a_2 x_{2i}$



=> to represent a point on a line  
$y$ is a dot product to matrix $x$ & $a$ i.neural networke

$\bar y $ =X $\bar a $ 

 
# Create the line


```python
%matplotlib inline
from fastai.basics import *
```


```python
n = 100
x = torch.ones(n,2) 
x[:,0].uniform_(-1.,1)
x[:5]
```



```console
    tensor([[-0.5704,  1.0000],
            [-0.4755,  1.0000],
            [ 0.5086,  1.0000],
            [-0.3661,  1.0000],
            [ 0.4988,  1.0000]])
```



```python
a = tensor(3.,2); a
```



```console
    tensor([3., 2.])
```



```python
y = x@a #+ torch.rand(n)
plt.scatter(x[:,0], y);
```


![png](/assets/images/2019-05-25/Simple_neural_network_example_files/Simple_neural_network_example_6_0.png)


In NN we have only `x & y`. `a` is not available to us and we have to predict `a`. 

To start calculation we randomly initialize `a`


```python
def mse(y_hat, y): 
    return ((y_hat-y)**2).mean()


a = nn.Parameter(torch.rand(2)); a

```



```console
    Parameter containing:
    tensor([0.5055, 0.0874], requires_grad=True)
```



```python
def update():
    y_hat = x @ a
    loss = mse(y_hat, y)
    if t%10==0:
        print(loss)
    loss.backward()
    # torch.no_grad() set requires_grad flag to false
    # requires_grad means this layer is available for training
    with torch.no_grad():
        # the gradient is showing where the next value of a should reside.
        # we subtract it from previous value.
        a.sub_(lr * a.grad) 
        a.grad.zero_()
```


```python
lr = 1e-1
for t in range(100):
    update()
```
```console
    tensor(5.4649, grad_fn=<MeanBackward0>)
    tensor(0.5915, grad_fn=<MeanBackward0>)
    tensor(0.1551, grad_fn=<MeanBackward0>)
    tensor(0.0434, grad_fn=<MeanBackward0>)
    tensor(0.0122, grad_fn=<MeanBackward0>)
    tensor(0.0034, grad_fn=<MeanBackward0>)
    tensor(0.0010, grad_fn=<MeanBackward0>)
    tensor(0.0003, grad_fn=<MeanBackward0>)
    tensor(7.5263e-05, grad_fn=<MeanBackward0>)
    tensor(2.1110e-05, grad_fn=<MeanBackward0>)
```


```python
a
```




    Parameter containing:
    tensor([2.9956, 1.9999], requires_grad=True)



We can the value of predicted `a` is close to original `[3,2]`


```python
plt.subplot(1, 2, 1)
plt.scatter(x[:,0],y, c='b')
plt.title("Original")
plt.subplot(1, 2, 2)
plt.scatter(x[:,0],x@a, c='r');
plt.title("Calculated")
```




    Text(0.5, 1.0, 'Calculated')




![png](/assets/images/2019-05-25/Simple_neural_network_example_files/Simple_neural_network_example_13_1.png)


Thanks https://fast.ai for this example.


```

```


```

```
