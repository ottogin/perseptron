from libc.math cimport exp
from libc.stdlib cimport calloc, free

import copy
import os.path
from decimal import *

cdef int num = 0
#etta - скорость обучения сети, менялась во время обучения
cdef float etta


#бинарная сигмоидальная функция с областью значений в ( 0, 1 )
#наиболее часто употребляемая в алгоритмах обратного распространения функция активации
#так как является непрерывной, дифференциируемой, монотонной и имеет простую производную
cdef double f_bin(double a):
	return 1 / (1 + exp(-a))


#сам класс нейросети
cdef class neuronweb:

	def __init__(self, tn_layers, tlayers, tn_features):
		self.n_layers = tn_layers

		#layers_init
		#Массив, содержащий количество нейронов в каждом слое
		#По сути равен  [n_features, n_actions, n_hidden]
		self.layers = <int*> calloc(tn_layers + 1, sizeof(int))
		self.layers[0] = tn_features
		for i in range(tn_layers):
			self.layers[i + 1] = tlayers[i]

		#ans init - array of state-vecotrs
		#двумерный массив данных каждого нейрона
		self.ans = <double**> calloc(tn_layers + 1, sizeof(double*))
		for i in range(self.n_layers + 1):
			self.ans[i] = <double*> calloc(self.layers[i], sizeof(double))

		#coefs init - array of coefs for each neuron
		#трехмерный массив весов связей
		#именно его будем изменять при обучении нейросети
		self.coefs = <double***> calloc(tn_layers, sizeof(double**))
		for i in range(tn_layers):
			self.coefs[i] = <double**> calloc(self.layers[i + 1], sizeof(double*))
			for j in range(self.layers[i + 1]):
				self.coefs[i][j] = <double*> calloc(self.layers[i] + 1, sizeof(double))

        #delta init - array of mistakes for each neuron
        #массив ошибок для градиентного спуска
		self.delta = <float**> calloc(tn_layers + 1, sizeof(float*))
		for i in range(self.n_layers + 1):
			self.delta[i] = <float*> calloc(self.layers[i], sizeof(float))



    #метод чтения коэфициентов из файла
    #параметр suf нужен, для функии random, чтобы считывать не из основного файла, а из измененного ф-ей sim_salabim
	def read_coefs(self,  suf = ""):
		for l in range(self.n_layers):
			name = "Data/layer" + str(l + 1) +".coefs" + suf
			if os.path.isfile(name):
				f = open(name, "r")
				for n in range(self.layers[l+1]):
					for c in range(self.layers[l]+1):
						self.coefs[l][n][c] = Decimal(f.readline())
				f.close()


    #метод записи коэфициентов в файл
    def write_coefs(self):
		for l in range(self.n_layers):
			name = "Data/layer" + str(l + 1) +".coefs"
			f = open(name, "w")
			for n in range(self.layers[l+1]):
				for c in range(self.layers[l]+1):
					f.write("%.40f\n" % self.coefs[l][n][c])
			f.close()


    #Каждый нейрон, получив сигнал, широковещательно(т.е. всем нейронам следующего слоя)
    #распространяет резульат активационной функции,
    #которая вызывается от суммы линейной комбинации полученного сигнала с коэффициэнтами, из массива coefs
    #Выходным сигналом сети, т.е ответом является выходной сигнал второго слоя нейронов
    cdef double* answer(self, float* state):
		cdef double summ
		for i in range(self.layers[0] + 1):
			self.ans[0][i] = state[i]
		for l in range(self.n_layers):
			for n in range(self.layers[l + 1]):
				summ = self.coefs[l][n][self.layers[l]]
				for c in range(self.layers[l] + 1):
					summ = summ + self.ans[l][c] * self.coefs[l][n][c]
				self.ans[l + 1][n] = f_bin(summ)
		return self.ans[self.n_layers]


    #основной метод обучения
    #здесь реализовано обратное распространение ошибки
	cdef void gradient_down(self, float* state, float* good_answer):
	    global num
		for l in range(self.n_layers, 0, -1):
			for n in range(self.layers[l]):
			    #Заранее вычисляем производную биполярного сигмоида
				self.delta[l][n] = self.ans[l][n] * (1 - self.ans[l][n])
				if (l == self.n_layers):
				    #Вычисляем разницу выходных параметров и правильного ответа
					self.delta[l][n] = self.delta[l][n] * good_answer[n] - self.ans[l][n]
				else:
					summ = 0
					for i in range(self.layers[l+1]):
                        #Получаем ошибки со всех нейронов,
                        #стоящих после вычисляемого сло
						summ = summ + self.coefs[l][i][n] * self.delta[l+1][i]
                    #Умножаем полученную суммарную ошибку на производную б.с.
					self.delta[l][n] = self.delta[l][n] * summ
				for c in range(self.layers[l-1]):
					#Записываем новые коэффициенты, учитывая скорость обучения
					self.coefs[l-1][n][c] = self.coefs[l-1][n][c] + etta * self.delta[l][n] * self.ans[l-1][c]
				self.coefs[l-1][n][self.layers[l-1]] = self.coefs[l-1][n][self.layers[l-1]] + etta * self.delta[l][n]



    #деструктор
	cdef void destroy(self):
		for i in range(self.n_layers + 1):
			free(self.ans[i])
		free(self.ans)
		for i in range(self.n_layers):
			for j in range(self.layers[i + 1]):
				free(self.coefs[i][j])
			free(self.coefs[i])
        for i in range(self.n_layers + 1):
			free(self.delta[i])
		free(self.delta)
		free(self.layers)
		free(self.coefs)
