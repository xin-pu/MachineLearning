'''
Created on Jul 16, 2019

@author: lena.li


'''

from scipy.interpolate import interp1d
import  numpy as np
# import matplotlib.pyplot as plt

x=np.linspace(0,10,num=11,endpoint=True)
y=np.cos(-x*2/9.0)
print(x)
print(y)
f=interp1d(x,y)
f2=interp1d(x,y,kind='cubic')


xnew=np.linspace(0,10,num=41,endpoint=True)
ynew=f2(xnew)
print(xnew)
print(ynew)

# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()

