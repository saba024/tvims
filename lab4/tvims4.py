from numpy import random  
from collections import Counter
import matplotlib.pyplot as plt     
import math
import pandas as pd
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import scipy.integrate as integrate

all_intervals = []
all_intervals1 = []

def random_variable(n):
	a = 0
	b = math.pi
	y0 = 0.5

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
	return Y

def count_expected_value(Y, n):
	expected_value = sum(Y) / n
	return expected_value

def count_variance(Y, n, expected_value):
	variance = 0
	for i in Y:
		variance += (i - expected_value)**2
	variance = variance / (n - 1)
	return variance

def find_intervals(t_g, expected_value, variance, n):
	intervals = []
	for t in t_g:
		intervals.append((expected_value - np.sqrt(variance) * t / np.sqrt(n - 1),expected_value + np.sqrt(variance) * t / np.sqrt(n - 1)))
	return intervals

def search_intervals(n, variance, gamma):
	intervals = []
	chi_mass = sts.chi2(n - 1)
	array = chi_mass.rvs(100000)
	xi_plus = []
	xi_minus = []
	for i in gamma:
		temp = sts.mstats.mquantiles(array, prob=[(1-i)/2, (1+i)/2])
		xi_plus.append(temp[0])
		xi_minus.append(temp[1])
	for i in range(3):
		intervals.append(((n-1) * variance / xi_minus[i], (n - 1) * variance / xi_plus[i]))
	return intervals

def count_theor_expected_value():
	MO_teor = integrate.quad(lambda x: (2 * x) / (math.pi * math.sqrt(1 - pow(x, 2))), 0, 1)[0]
	return MO_teor

def count_theor_variance(expected_value):
		D_teor = integrate.quad(lambda x: 2 * math.pow(x, 2) / (math.pi * math.sqrt(1 - pow(x, 2))), 0, 1)[0] - expected_value**2
		return D_teor

def show(gamma, intervals, intervals2):
	plt.plot(gamma, [interval[1] - interval[0] for interval in intervals])
	plt.plot(gamma, [interval[1] - interval[0] for interval in intervals2])
	plt.xlabel("$\gamma$")
	plt.ylabel("Interval value")
	plt.show()

def draw(intervals):
	plt.plot(gamma, [interval[1] - interval[0] for interval in intervals])
	plt.xlabel("$\gamma$")
	plt.ylabel("Interval value")
	plt.show()

def task1(n, t_g, gamma):

	Y = random_variable(n)
	expected_value = count_expected_value(Y, n)
	print("M = ", expected_value)
	
	variance = count_variance(Y, n, expected_value)
	print("D = ", variance)

	intervals = find_intervals(t_g, expected_value, variance, n)
	print(intervals)
	
	draw(intervals)

	MO_teor = count_theor_expected_value()

	D_teor = count_theor_variance(MO_teor)

	intervals2 = find_intervals(t_g, MO_teor, D_teor, n)
	print(intervals2)
	all_intervals.append(intervals2)

	str = "n = " + n.__str__()+ " M = " + expected_value.__str__() + " D = " + variance.__str__()
	draw(intervals2)

	show(gamma, intervals, intervals2)

def task2(n, t_g, gamma):
	Y = random_variable(n)
	MO = count_expected_value(Y, n)
	print('MO = ', MO)
	Disp = count_variance(Y, n, MO)
	print('D = ', Disp)
	intervals = search_intervals(n, Disp, gamma)
	print(intervals)

	draw(intervals)

	MO_teor = count_theor_expected_value()
	Disp1 = count_variance(Y, n, MO_teor)
	print("D = ", Disp)

	intervals2 = search_intervals(n + 1, Disp1, gamma)
	print(intervals2)
	all_intervals1.append(intervals2)

	draw(intervals2)
	show(gamma, intervals, intervals2)



N = [30, 50, 70, 100, 150]
t_g = [1.73, 2.093, 2.861]
gamma = [0.9, 0.95, 0.99]

task1(20, t_g, gamma)
task1(50, t_g, gamma)
task1(70, t_g, gamma)
task1(100, t_g, gamma)
task1(150, t_g, gamma)

task2(20, t_g, gamma)
task2(50, t_g, gamma)
task2(70, t_g, gamma)
task2(100, t_g, gamma)
task2(150, t_g, gamma)

plt.plot(N, [(interv[0][1] - interv[0][0]) for interv in all_intervals])
plt.plot(N, [(interv[1][1] - interv[1][0]) for interv in all_intervals])
plt.plot(N, [(interv[2][1] - interv[2][0]) for interv in all_intervals])
plt.show()

plt.plot(N, [(interv1[0][1] - interv1[0][0]) for interv1 in all_intervals1])
plt.plot(N, [(interv1[1][1] - interv1[1][0]) for interv1 in all_intervals1])
plt.plot(N[:-1], [(interv1[-1][1] - interv1[-1][0]) for interv1 in all_intervals1[:-1]])
plt.show()

