#cdef float f_bin_pol(float a)
#cdef float f_bin(float a)

cdef class neuronweb:
	cdef int n_layers
	cdef int* layers
	cdef float** ans
	cdef float*** coefs
	cdef float* answer(self, float* state)

