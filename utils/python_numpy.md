
# Numpy and Scipy
```python
import numpy as np
import scipy as sp
from scipy import (signal, fft)
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

## np.fft

`rfft` function computes the one-dimensional discrete Fourier Transform for real input, of which `irfft` function is inverse.

**Note**: when the DFT is computed for purely real input, the output is *Hermitian-symmetric*, i.e. the negative frequency terms are just the complex conjugates of the corresponding positive-frequency terms, and the negative-frequency terms are therefore redundant. **This function does not compute the negative frequency terms, and the length of the transformed axis of the output is therefore `n//2 + 1`.**

When `A = rfft(a)` and fs is the sampling frequency, `A[0]` contains the zero-frequency term 0*fs, which is real due to Hermitian symmetry. If *n* is even, `A[-1]` contains the term representing both positive and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely real. If *n* is odd, there is no term at fs/2; `A[-1]` contains the largest positive frequency (fs/2*(n-1)/n), and is complex in the general case.

```python
# a: array like
# n: number of points to do FFT
np.fft.rfft(a, n=None)

# a: array like
# n: lenght of the output 
np.fft.irfft(a, n=None)
```

