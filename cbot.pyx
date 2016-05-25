import interface as bbox
cimport interface as bbox
import numpy as np
import random
from libc.math cimport fabs

import neuronweb
cimport neuronweb

cdef neuronweb.neuronweb nw
cdef int n_features = 36, n_actions = 4
cdef int n_hidden = (n_features - n_actions) / 2, n_layers = 2

cdef int lastact = 0

cdef int eps_processing(double* ans):
	action = 0
	best_val = ans[0]
	for act in range(n_actions):
		if ans[act] > best_val:
			action = act
			best_val = ans[act]
	return action

cdef int eps_processing1(double* ans):
	cdef double eps = 1E-4
	action = 0
	best = ans[0]
	for act in range(n_actions):
		if ans[act] > best:
			action = act
			best = ans[act]
	subaction = 0
	subbest = ans[0]
	for act in range(n_actions):
		if ans[act] > subbest and act != action:
			subaction = act
			subbest = ans[act]
	#if fabs(subbest - best) / best < eps:
		#return random.choice([action, subaction])
	return action

cdef int get_action_by_state(float* state):
	global lastact
	cdef double* ans = nw.answer(state)
	cdef double best_val = ans[0]
	lastact = eps_processing(ans)
	return lastact
	

def prepare_bbox(name):
	global n_features, n_actions, last_level, n_hidden
	if bbox.is_level_loaded() and last_level == name:
		bbox.reset_level()
	else:
		bbox.load_level(name, verbose = 1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
		n_hidden = (n_features - n_actions) / 2
	last_level = name
 
def run_level(name):
	cdef:
		float* state
		int action, has_next = 1
	prepare_bbox(name)
	while has_next:
		state = bbox.c_get_state()
		action = get_action_by_state(state)
		has_next = bbox.c_do_action(action)
	bbox.finish(verbose = 0)
	return bbox.get_score()
 
def run():
	global nw
	nw = neuronweb.neuronweb(n_layers, [n_hidden, n_actions], n_features)
	nw.read_coefs();
	train = run_level("levels/train_level.data")
	test = run_level("levels/test_level.data")
	print("Train:" + str(train) + " Test:" + str(test)) 
	nw.destroy()

