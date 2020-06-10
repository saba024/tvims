from numpy import random  
from collections import Counter
import matplotlib.pyplot as plt     
import math
import pandas as pd
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np



def Fn(N, size):
	Fn = [0]
	for index in range(0, len(N)):
		Fn.append(Fn[index] + N[index] / size)
	return Fn

def get_N(Y):
	N = [1]
	for index in range(1, len(Y)):
		if Y[index - 1] == Y[index]:
			Y.remove(Y[index])
			N[len(N) - 1] += 1
		else:
			N.append(1)
	return N

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



N = get_N(Y)
F = Fn(N, n)
Y = np.array(sort_Y)
F = np.array(F[1:])
plt.figure()
plt.title("Графики эмпирической функции")
plt.xlabel("Y")
plt.ylabel("Fn(Y)")
plt.plot(sort_Y, F, 'wo', mec='black')
plt.plot([0, sort_Y[0]], [0, 0])
plt.plot([Y[0], sort_Y[0]], [0, F[0]])
for index in range(0, len(sort_Y) - 1):
	plt.plot([sort_Y[index], sort_Y[index + 1]], [F[index], F[index]])
	plt.plot([sort_Y[index + 1], sort_Y[index + 1]], [F[index], F[index + 1]])
plt.plot([sort_Y[len(Y) - 1], 1], [1, 1])
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
	f_y.append((2 * math.asin(xi)) / math.pi)

plt.plot(x_theor, f_y, label='Theoretical distribution function')
plt.xlabel('x')
plt.ylabel('F(y)')
plt.legend()
plt.show()

