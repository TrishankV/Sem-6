import numpy as np
import matplotlib as mtp
import math
import pandas as pd
n =11
x1 = [3.1,5.31,5,1.185,6.1,7.4,6.4,6.84,3.24,6.84,3.66]
x2 = [1.231,2.312,3.1234,1.2,3.13,4.12,4.12,4.124,1.24,4.1,5.5]
y = [0,0,0,1,1,0,0,1,1,1,1]
print(f'X1 = {x1}\nX2 = {x2}\ny = {y}')
X1 = [3.1, 5.31, 5, 1.185, 6.1, 7.4, 6.4, 6.84, 3.24, 6.84, 3.66]
X2 = [1.231, 2.312, 3.1234, 1.2, 3.13, 4.12, 4.12, 4.124, 1.24, 4.1, 5.5]
y = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
# len(y)
# len(x1)
# len(x2)
# 11
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
print(f'X1 = {x1.reshape(-1,1)} , X2 = {x2.reshape(-1,1) } , y = {y.reshape(-
1,1)}')
X1 = [[3.1 ]
[5.31 ]
[5. ]
[1.185]
[6.1 ]
[7.4 ]
[6.4 ]
[6.84 ]
[3.24 ]
[6.84 ]
[3.66 ]] , X2 = [[1.231 ]
[2.312 ]
[3.1234]
[1.2 ]
[3.13 ]
[4.12 ]
[4.12 ]
[4.124 ]
[1.24 ]
[4.1 ]
[5.5 ]] , y = [[0]
[0]
[0]
[1]
[1]
[0]
[1]
[1]
[1]
[1]
[1]]
b0, b1 , b2 = 0 , 0 , 0
s = 0.3
p = []
pc = []
for i in range(0, 11):
z = b0 + b1 * x1[i] + b2 * x2[i]
f = 1 / (1 + math.e**(-z))
p.append(f)
b0 = b0 + (s * ((y[i] - p[i])) * p[i] * (1 - p[i]))
b1 = (b1 + (s * ((y[i] - p[i])) * p[i] * (1 - p[i]))) * x1[i]
b2 = (b2 + (s * ((y[i] - p[i])) * p[i] * (1 - p[i]))) * x2[i]
if p[i] > s :
  pc.append(1)
else :
  pc.append(0)
# /var/folders/ys/f1dtzjmj589424_wn9_wp8t80000gn/T/ipykernel_24373/200101412.py:3
# : RuntimeWarning: overflow encountered in scalar power
f = 1 / (1 + math.e**(-z))
print(y.reshape([-1,1]) , np.reshape(p,[-1,1]) , np.reshape(pc , [-1,1]) , end=" ")

print(y)
print(p)
print(pc)

asd = pd.DataFrame(data = { 'X1' : x1 ,'x2' : x2 ,'y' : y,'prediction' : p ,'prediction class ' : pc})
print(asd)
# X1 x2 y prediction prediction class
# 0 3.100 1.2310 0 5.000000e-01 1
# 1 5.310 2.3120 0 3.183174e-01 1
# 2 5.000 3.1234 0 1.510054e-02 0
# 3 1.185 1.2000 1 7.048290e-03 0
# 4 6.100 3.1300 1 6.024425e-13 0
# 5 7.400 4.1200 0 2.003287e-88 0
# 6 6.400 4.1200 1 0.000000e+00 0
# 7 6.840 4.1240 1 0.000000e+00 0
# 8 3.240 1.2400 1 0.000000e+00 0
# 9 6.840 4.1000 1 0.000000e+00 0
# 10 3.660 5.5000 1 0.000000e+00 0
