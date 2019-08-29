#  The linalgpackage
#  The fftpackage
#  Random numbers
#  Continuous and discrete distributions

import numpy
import matplotlib.pyplot as plt

A = numpy.mat("0 1 2;1 0 3;4 -3 8")
print(A)


# Invert the matrix:
inverse = numpy.linalg.inv(A)
print(inverse)

print(A*inverse)


# creating a matrix
# 1.  Which function can create matrices?
# a.  array
# b.  create_matrix
# c.  mat
# d.  vector

A = numpy.mat("1 -2 1;0 2 -8;-4 5 9")

b = numpy.array([0, 8, -9])

# Call the solve function: Solve this linear system with the solvefunction:
x = numpy.linalg.solve(A, b)
x=numpy.linalg.solve(A,b)
print(x)
print(numpy.dot(A , x))


# Calculate eigenvalues with the eig function:

A = numpy.mat("3 -2;1 0")
print(numpy.linalg.eigvals(A))

eigenvalues, eigenvectors = numpy.linalg.eig(A)
print(eigenvectors)


#pseudo inverse:
A = numpy.mat("4 11 14;8 7 -2")
pseudoinv = numpy.linalg.pinv(A)
print(A * pseudoinv)



# Fast Fourier transform

x = numpy.linspace(0, 2 * numpy.pi, 30)
wave = numpy.cos(x)

transformed = numpy.fft.fft(wave)
d = numpy.fft.ifft(transformed)
plt.plot(transformed)
plt.plot(d)
plt.show()


# Apply the inverse transform: Apply the inverse transform with the ifftfunction. It 
# should approximately return the original signal.


# Random numbers