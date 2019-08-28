import numpy
from numpy import dtype

def numpysum(n):
    a = numpy.arange(n) ** 2
    b = numpy.arange(n) ** 3
    c = a + b
    return c


print(numpysum(3))


# 1. What does arange(5) do?
# Creates a Python list of 5 elements with values 0 to 4.

a = numpy.arange(5)
print(a)
print(a.shape)

print(a[1])

# Type Character
# code
# integer i
# Unsigned integer u
# Single precision float f
# Double precision float d
# bool b
# complex D
# string S
# unicode U
# Void V
print(numpy.arange(7, dtype='f8'))
dtype('d')
t= dtype('float64')
print(t.char)


#One-dimensional slicing and indexing
a=numpy.arange(9)
print(a[3:7])
print(a[:7:2])
print(a[::-1])

b=numpy.arange(24).reshape(2,3,4)
print(b)
print(b[:,0,0])
print(b[1])
print(b[1,::-1,-1])
print(b[1,::2,-1])

#If we want to select all the rooms on both floors that are in
# the second column, regardless of the row, we will type the following code snippet:
print()
print(b[...,1])
print(b[:,:,1])
print(b[:,1])

newb=b.flatten()
c=numpy.ravel(newb)
print(c)

#Setting the shape with a tuple
c.shape=(6,4)
print(c)

#In linear algebra, it is common to transpose matrices. We can do that 
# too, by using the following code:
d=c.transpose()
print(c)

d.resize((2,12))
print(d)