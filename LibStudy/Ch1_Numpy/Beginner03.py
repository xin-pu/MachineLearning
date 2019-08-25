from numpy import *

a = arange(24).reshape(2, 12)
print(a)

print(a.ndim)
print(a.size)
print(a.itemsize)
print(a.nbytes)

a.resize(6,4)
print(a)
print(a.T)

print(a.tolist())

b = array([1.j + 1, 2.j + 3])
print(b.real)
print(b.imag)
