from uncvalue import *
import numpy as np 
import matplotlib.pyplot as plt

###########################################
#			 Basic operations			  #
###########################################

a = Value(3.12, 0.52)
b = Value(2.569, 0.198)
c = Value(0.00238, 0.0005498)
# call val()/unc() to get the value/uncetainty of a Value
print('Value = %.2f +/- %.2f' % (val(a), unc(a)))
print('Value =', c)

# perform any operation always using numpy library
# operations made with python math library will not work
print('a + b =', a + b)
print('a*sin(b) =', a * np.sin(b))

###########################################
#		   		  Lists					  #
###########################################
def random_list(lth):
	return [Value(10*np.random.rand(), np.random.rand()) for _ in range(lth)]

A = random_list(100)
# Evaluate the mean value
print('List mean =', np.mean(A))

# print the sine of elements 60 to 64
print('sin(A) =', np.sin(A)[60:65])

###########################################
#		   Matrix multiplication		  #
###########################################

def random_matrix(m, n):
	# initialize the empty array with object type
	# this has to be done because it will contain Value(s)
	# not primitive number types like floats
	M = np.empty((m, n), dtype=object)

	for i in range(m):
		for j in range(n):
			M[i,j] = Value(np.random.rand(), np.random.rand()/(j+1))

	return M

H, M = random_matrix(5,3), random_matrix(3,2)
print('Matrix product =\n', np.dot(H, np.tanh(M)))

###########################################
#			  Plot examples				  #
###########################################

def generate_random_data_1(lth):
	# generates a value list
	return Value(np.random.rand(lth, 2), np.random.rand(lth, 2)/10)

def generate_random_data_2(lth):
	# generates two list of values
	X, Y = [], []
	for _ in range(lth):
		X.append(Value(np.random.rand(), np.random.rand()/10))
		Y.append(Value(np.random.rand(), np.random.rand()/20))

	return np.array(X), np.array(Y)

rand_data_1 = generate_random_data_1(10)

rand_data_2 = generate_random_data_2(10)

rand_data_3 = rand_data_2[0]**2 * np.log(rand_data_2[1]) + Value(2.02, 0.25)

plt.figure()
plt.errorbar(val(rand_data_1)[:,0], val(rand_data_1)[:,1], 
	xerr=unc(rand_data_1)[:,0], yerr=unc(rand_data_1)[:,1],
	ls='None', c='b', marker='None', label='1')

plt.errorbar(val(rand_data_2[0]), val(rand_data_2[1]), 
	xerr=unc(rand_data_2[0]), yerr=unc(rand_data_2[1]),
	ls='None', c='r', marker='None', label='2')

plt.errorbar(val(rand_data_2[0]), val(rand_data_3), 
	xerr=unc(rand_data_2[0]), yerr=unc(rand_data_3),
	ls='None', c='g', marker='None', label='3')

plt.legend()
plt.show()