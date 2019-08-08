

[TOC]

# Training Deep Neural Nets

## Gradient Vanish/Explode

More generally, deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds. This problem was one of the reasons why deep neural networks were mostly abandoned for a long time.

### Weights Initialization

the connection weights must be initialized randomly as described in the following equation, where n~inputs~ and n~outputs~ are the number of input and output connections for the layer whose weights are being initialized. This initialization strategy is often called *Xavier initialization* or sometimes *Glorot initialization*.
$$
Normal \ distribution \ with \ mean \ 0 \ and \ standard \ deviation \\
\sigma = \sqrt{2\over{n_{inputs} - n_{outputs}}} \\
or \ a \ uniform \ distribution \ between \ -r \ and \ r, \ with \\
r = \sqrt{6\over{n_{inputs}+n_{outputs}}}
$$


The initialization strategy for ReLU activation function (incl. its variants) is sometimes called *He initialization*.

![]

### Nonsaturating Activation Functions

The vanishing/exploding gradient problems were in part due to a poor choice of activation function. In particular, the ReLU function behave much better than logistic activation function, mostly because it does not saturate for positive values (and also because it is quite fast to compute).

**Unfortunately, ReLU function is not perfect. It suffers from a problem known as *dying ReLUs*: during training, some neurons effectively die, meaning they stop outputting anything other than 0.** In some cases, you may find that half of your network’s neurons are dead, especially if you used a large learning rate. During training, if a neuron’s weights get updated such that the weighted sum of the neuron’s inputs is negative, it will start outputting 0. When this happen, the neuron is unlikely to come back to life since the gradient of the ReLU function is 0 when its input is negative.

To solve this problem, you may want to use a variant of the ReLU function, such as *leaky ReLU*.
$$
LeakyReLU_\alpha(z) = max(\alpha z, z)
$$
The hyperparameter alpha defines how much the function 'leaks'.





### Batch Normalization



### Gradient Clipping



## Avoid Overfitting















