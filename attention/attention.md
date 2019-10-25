

Encode each word in the sentence into a vector

When decoding, perform a linear combination of these vectors, weighted by "attention weights"

Use this combination in picking the next word 



Attention Score Function

*q* is the query and *k* is the key.

Multi-layer Perceptron:

$a(q,k) = \omega^{T}_{2}tanh(W_1[q;k])$

Flexible, often very good with large data.

seq2seq from TensorFlow