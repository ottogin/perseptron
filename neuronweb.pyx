import copy
import os.path 

cdef extern from "/home/ottoin/blackbox/neuronweb/help.c":
	float f_bin_pol(float a)
	float f_bin(float a)

from cymem.cymem cimport Pool
cdef Pool mem = Pool() 

cdef class neuronweb:
	def __init__(self, tn_layers, tlayers, tn_features):
		global mem
		self.n_layers = tn_layers
		#layers_init
		self.layers = <int*> mem.alloc(tn_layers + 1, sizeof(int))
		self.layers[0] = tn_features
		for i in range(tn_layers):
			self.layers[i + 1] = tlayers[i]
		#ans init - array of state-vecotrs 
		self.ans = <float**> mem.alloc(tn_layers + 1, sizeof(float*))
		for i in range(self.n_layers+1):
			self.ans[i] = <float*> mem.alloc(self.layers[i], sizeof(float))
		#coefs init - array of coefs for each neuron
		self.coefs = <float***> mem.alloc(tn_layers, sizeof(float**))
		for i in range(tn_layers):
			self.coefs[i] = <float**> mem.alloc(self.layers[i + 1], sizeof(float*))
			for j in range(self.layers[i + 1]):
				self.coefs[i][j] = <float*> mem.alloc(self.layers[i] + 1, sizeof(float))

	def read_coefs(self):
		for l in range(self.n_layers):
			name = "Data/layer" + str(l)+".coefs"
			if os.path.isfile(name): 
				f = open(name, "r")        
				for n in range(self.layers[l+1]):
					for c in range(self.layers[l]+1):    
						self.coefs[l][n][c] = float(f.readline())  
				f.close() 

	cdef float* answer(self, float* state):
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
