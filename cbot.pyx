import interface as bbox
cimport interface as bbox
import numpy as np

import neuronweb
cimport neuronweb

cdef neuronweb.neuronweb nw
cdef int n_features = 36, n_actions = 4
cdef int n_hidden = 16, n_layers = 2
cdef float eps = 0.002

cdef int get_action_by_state_fast(float* state):
	cdef float best_val = -1e9
	cdef float* ans = nw.answer(state)
	action_to_do = 0
	for act in range(n_actions):
		if ans[act] > best_val:
			action_to_do = act
			best_val = ans[act]
	# TODO: разность < eps ==> ответить как в прошлый раз
	return action_to_do

def prepare_bbox(name):
	global n_features, n_actions, last_level
	if bbox.is_level_loaded() and last_level == name:
		bbox.reset_level()
	else:
		bbox.load_level(name, verbose = 1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
	last_level = name
 
def run_level(name):
	cdef:
		float* state
		int action, has_next = 1
	prepare_bbox(name)
	while has_next:
		state = bbox.c_get_state()
		action = get_action_by_state_fast(state)
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

