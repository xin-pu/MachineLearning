from numpy import *

a = arange(9).reshape(3, 3)
b = a*2


# Horizontal stacking
print(hstack((a, b)))
print(concatenate((a, b), axis=1))

# Vertical stacking
print(vstack((a, b)))
print(concatenate((a, b), axis=0))

# Depth stacking
print(dstack((a, b)))

# Column stacking:
print(column_stack((a, b)))

# Row stacking:
print(row_stack((a, b)))

# Horizontal splitting
print(a)
print(hsplit(a, 3))
print(split(a, 3, axis=1))

# Vertical splitting
print(vsplit(a, 3))
print(split(a, 3, axis=0))

# Depth-wise splitting

