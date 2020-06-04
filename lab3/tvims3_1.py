from numpy import random  
from collections import Counter
import matplotlib.pyplot as plt     
import math
import pandas as pd
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

a = 0
b = math.pi
y0 = 0.5

n = 30
X = []
Y = []
r = sts.uniform()
xi = r.rvs(size=n)

for i in range(n):
	x = xi[i]*(b - a) + a
	X.append(x)
	y = math.sin(x)
	Y.append(y)

Y.sort()
var_series = pd.DataFrame(data={'$Y_i$': Y})
print(var_series.T)

emp_dist_func = ECDF(Y)
print(emp_dist_func.y)
f_y = []
x_theor = np.linspace(0, 0.99, 30)
for xi in x_theor:
	f_y.append(2 / (math.pi * math.sqrt(1 - math.pow(xi, 2)))) # теоретическая функция распределения
plt.plot(x_theor, f_y, label='Theoretical distribution function')
plt.step(emp_dist_func.x, emp_dist_func.y, label='Empirical distribution function')
plt.ylabel('F(y)')
plt.xlabel('x')
plt.legend()
plt.show()

d_plus = []
d_minus = []
for i in range(n - 1):
    d_plus.append(abs((i + 1) / n - (2 * math.asin(Y[i])) / math.pi))
    d_minus.append(abs(i / n - (2 * math.asin(Y[i])) / math.pi))

d = max(max(d_plus), max(d_minus))
print("d = ", d)

lambd = d * np.sqrt(n)
print("lambda = ", lambd)
