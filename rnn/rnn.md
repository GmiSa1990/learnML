[TOC]

## Basic RNN

<img src="c:/python/learnML/images/rnn.PNG" style="zoom:75%;" />

Figure: A recurrent neural network and the unfolding in time of the computation involved in its forward computation.

$x_t$ is the input at time step $t$. 

$s_t$ is the **hidden state** at time step $t$. It's the "memory" of the network. $s_t$ is calculated with the following formula:

$$
s_t = f(Ux_t + Ws_{t-1})
$$


the function $f$ usually is a nonlinearity such as tanh or ReLU.

$o_t$ is the output at time step $t$, which is calculated with the formula:

$$
o_t = g(Vs_t)
$$


Given the dimension of matrix U is [n, m], W is [n, n], then:

 <img src="c:/python/learnML/images/rnn_state_calc.PNG"  />



## Long Short-Term Memory (LSTM)

+ [Reference Post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

+ One input and One output here. 
  ![lstm](c:/python/learnML/images/lstm_diagram.PNG)

  ![](c:/python/learnML/images/lstm_diagram_2.PNG)

+ The first step in LSTM is to decide what information we're going to throw away from the cell state. This decision is made by a sigmoid layer called the "forget gate layer".

  ![](c:/python/learnML/images/lstm_diagram_3.PNG)

+ Next step is to decide what new information we're going to store in the cell state. This has two parts.

  ![](c:/python/learnML/images/lstm_diagram_4.PNG)

+ Get new cell state via dropping the information from old cell state and adding new information to new cell state

  ![](c:/python/learnML/images/lstm_diagram_5.PNG)

+ Finally, we need to decide what we're going to output.

  ![](c:/python/learnML/images/lstm_diagram_6.PNG)



## GRU

[Reference Post](https://d2l.ai/chapter_recurrent-neural-networks/gru.html)

<img src="c:/python/learnML/images/gru_diagram.PNG" style="zoom:75%;" />

+ Reset Gates and Update Gates

  A reset variable would allow us to control how much of the previous state we might still want to remember. Likewise, an update variable would allow us to control how much of the new state is just a copy of the old state.

  Reset gates help capture short-term dependencies in time series.

  Update gates help capture long-term dependencies in time series.

`kernel_regularizer` & `recurrent_regularizer` & `bias_regularizer` & `activity_regularizer`

[1](https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers)    [2](https://stats.stackexchange.com/questions/383310/difference-between-kernel-bias-and-activity-regulizers-in-keras)

"Here is the answer: I encountered a case where the weights of the net are small and nice, ranging between [-0.3] to [+0.3]. So, I really can't punish them, there is nothing wrong with them. A kernel regularizer is useless. However, the output of the layer is HUGE, in 100's.
Keep in mind that the input to the layer is also small, always less than one. But those small values interact with the weights in such a way that produces those massive outputs. Here I realized that what I need is an activity regularizer, rather than kernel regularizer. With this, I'm punishing the layer for those large outputs, I don't care if the weights themselves are small, I just want to deter it from reaching such state cause this saturates my sigmoid activation and causes tons of other troubles like vanishing gradient and stagnation."

`kernel_regularizer` & `kernel_constraint`

"Constraining the weight matrix directly is another kind of regularization. If you use a simple `L2 regularization` term you penalize high weights with your loss function. With this constraint, you regularize directly. As also linked in the `keras`code, this seems to work especially well in combination with a dropout layer. "

[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

In chapter 5.1, "One particular form of regularization was found to be especially useful for dropout -- constraining the norm of the incoming weight vector at each hidden unit to be upper bounded by a fixed constant *c*, in other words, if **w** represents the vector of weights incident on any hidden unit, the neural network was optimized under the constraint "

[A Gentle Introduction to Weight Constraints in Deep Learning](https://machinelearningmastery.com/introduction-to-weight-constraints-to-reduce-generalization-error-in-deep-learning/)

"Smaller weights in a neural network can result in a model that is more stable and less likely to overfit the training dataset, in turn having better performance when making a prediction on new data."

"Unlike weight regularization, a weight constraint is a trigger that checks the size or magnitude of the weights and scales them so that they are all below a pre-defined threshold. The constraint forces weights to be small and can be used instead of weight decay and in conjunction with more aggressive network configurations, such as very large learning rates."

[How to Reduce Overfitting Using Weight Constraints in Keras](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/)
