from numpy import random  
from collections import Counter
import matplotlib.pyplot as plt     
import math
import pandas as pd
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

def empiric_func(t):
	t_without_repeat = sorted(list(set(t)))
	t_without_repeat.insert(0, -float('Inf'))
	size = len(t)
	F_y = []
	p = 0
	for el in t_without_repeat:
		p += t.count(el)/size
		F_y.append(p)
	return list(t_without_repeat), F_y

# a = 0 b = pi x = r * (b - a) + a
a = 0
b = math.pi
y0 = 0.5

n = int(input())
X = []
Y = []
r = sts.uniform()
xi = r.rvs(size=n)

for i in range(n):
	x = xi[i]*(b - a) + a
	X.append(x)
	y = math.sin(x)
	Y.append(y)

print(Y)

sort_Y = sorted(Y)
emp_x, emp_y = empiric_func(sort_Y)
plt.step(emp_x, emp_y, label='Empirical distribution function', color='green')
plt.plot([emp_x[1]-0.5, emp_x[1]], [0,0] ,color='green')
plt.plot([emp_x[-1], emp_x[-1]+0.5], [1,1] , color='green')
plt.xlabel('x')
plt.ylabel('F(y)')
plt.legend()
plt.show()

# проверка с использованием встроенной функции
emp_dist_func = ECDF(Y)
plt.step(emp_dist_func.x, emp_dist_func.y, label='Empirical distribution function')
plt.xlabel('x')
plt.ylabel('F(y)')
plt.legend()
plt.show()

print("Вариационный ряд:")
data = {"Значение": sort_Y}
table = pd.DataFrame(data=data)
table.T
print(table)

f_y = []
x_theor = np.linspace(0, 0.99, 30)
for xi in x_theor:
	f_y.append(2 / (math.pi * math.sqrt(1 - math.pow(xi, 2))))

plt.plot(x_theor, f_y, label='Theoretical distribution function')
plt.xlabel('x')
plt.ylabel('F(y)')
plt.legend()
plt.show()

