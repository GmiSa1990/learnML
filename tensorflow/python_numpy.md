
# Numpy
```python
import numpy as np
```
## Basic Functions
```python
x = np.linspace(1.0, 10.0, 10)
# x = array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

y = x[:,np.newaxis]
# y = array([[ 1.],
#             [ 2.],
#             [ 3.],
#             [ 4.],
#             [ 5.],
#             [ 6.],
#             [ 7.],
#             [ 8.],
#             [ 9.],
#             [10.]])

# array creation
b = np.arange(10).reshape(2,5)
# array([[0, 1, 2, 3, 4],
#       [5, 6, 7, 8, 9]])

b.shape
#(5,2)
b.dtype.name
#'int32'
b.ndim
# 2
b.size
# 10

# shape manipulation
c = np.reshape(b, (2,5))


```