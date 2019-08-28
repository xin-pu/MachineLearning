import numpy as np

array_01 = np.array([1, 2, 3, 4, 5])
print(array_01)

print(np.zeros(5))
print(np.ones(5))
print(np.random.random(5))
print(np.ones((5,5)))

array_02=np.random.random((2,2))
print(array_02)
print(array_02[0][0])
print(array_02[0,:])

array_03=np.ones((2,2))
print(array_03+1)
print(array_03*2)

print(array_02*array_03)
print(array_02.dot(array_03))