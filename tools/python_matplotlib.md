

# Matplotlib
## pyplot
```python
import matplotlib.pyplot as plt
```
+ subplots

create a figure and a set of subplots(axes).
```python
import numpy as np
x = np.random.rand(1,5)
y = np.square(x)
z = np.sin(x)
fig, (ax1, ax2) = plt.subplots(1,2,sharex = True)
ax1.plot(x,y)
ax1.set_title('square(x)')
ax2.scatter(x,z)
plt.ion()  # Turn the interactive mode on. allow code to continue running after plt.show()
plt.show()  # show the figure
plt.pause(2) # pause for 2 seconds.
plt.clf()  # clear the figure
```

+ figure

create a **Figure** object
```python
fig = plt.figure('hello world')
ax = fig.add_subplot(1,1,1)
a = ax.plot(x,y)
b = ax.scatter(x,z)
plt.ion()
plt.show()
plt.pause(2)
ax.lines.remove(a[0]) # remove the line from the ax
plt.pause(2)
ax.collections.remove(b) # remove the scatter from the ax
plt.pause(2)
```