#cdef float f_bin_pol(float a)
#cdef float f_bin(float a)

cdef class neuronweb:
	cdef int n_layers
	cdef int* layers
	cdef double** ans
	cdef double*** coefs
	cdef double* answer(self, float* state)
	cdef void destroy(self)
	cdef void gradient_down(self, float* state, float* good_answer, float etta)
