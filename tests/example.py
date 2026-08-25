import ten

import numpy as np

x = ten.tensor([2, 2])
y = ten.tensor([2, 2])

print(x)
print(y)

x = ten.tensor((2, 2), ten.dtype.float32)
print(x)

# Set values
k = 0.
for i in range(2):
  for j in range(2):
    x[i,j] = k
    k += 1

# Print
print(x)

# Print
for i in range(2):
  for j in range(2):
    print(x[i,j])

# Row major tensor
y = ten.tensor.row_major((2,2))
print(y)

"""
# Ones
y = ten.ones((3, 3))
print(y)

# Zeros
z = ten.zeros((3,3))
print(z)

# fill
t = ten.fill((3,3), 9.9)
print(t)

# range
r = ten.arange((3,3), 10.)
print(r)

# linear
s = ten.linear((3,3), 1., 10.)
print(s)

# From numpy
x = np.arange(0, 9, dtype=np.float32).reshape(3, 3)
print(x)
y = ten.from_numpy(x)
print(y)

# To numpy
x = ten.arange((3,3))
print(x)
y = x.to_numpy()
print(y)
"""

