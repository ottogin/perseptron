import interface as bbox
cimport interface as bbox
import numpy as np
import time
import sys
from libc.math cimport fabs
import os.path
import neuronweb
cimport neuronweb
from decimal import *
import random as rand

cdef neuronweb.neuronweb nw

cdef int n_features = 36    #Объявляем размерность вектора состояния по дефолту (нулевой слой нейросети)
cdef int n_actions = 4      #количество возможных ответов(= количеству нейронов во втором слое нейросети)
cdef int n_hidden = (n_features - n_actions) / 2, n_layers = 2  #количество нейронов в первом слое нейросети

#глобальные переменные, необходимые функции train
cdef float last_score = 0
cdef int prev_act = 0



#выбираем лучшее действие по ответу нейросети
#в финальной версии программы решили оставить просто максимум по очкам
cdef int eps_processing(double* ans):
	action = 0
	best_val = ans[0]
	for act in range(n_actions):
		if ans[act] > best_val:
			action = act
			best_val = ans[act]
	return action


#Спрашиваем нейросеть
#Возвращаем результат работы eps_processing
cdef int get_action_by_state(float* state):
	global lastact
	cdef double* ans = nw.answer(state)
	cdef double best_val = ans[0]
	return eps_processing(ans)



def prepare_bbox(name):
	global n_features, n_actions, last_level, n_hidden
	if bbox.is_level_loaded() and last_level == name:
		bbox.reset_level()
	else:
		bbox.load_level(name, verbose = 0)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
		n_hidden = (n_features - n_actions) / 2
	last_level = name


#основной цикл прохождения уровня
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


#волшебная функция, изменяющая коэфициенты нейросети случайным образом
def sim_salabim():
	coefs = []
	for l in range(2):
		name = "Data/layer" + str(l + 1)+".coefs"
		if os.path.isfile(name):
			f = open(name, "r")
			for line in f.readlines():
				val = Decimal(line)
				if rand.random() > 0.995:
					if rand.random() > 0.01:
						coefs.append(- val * Decimal(rand.random()) * 2)
					else:
						coefs.append(  val * Decimal(rand.random()) * 2)
				else:
					coefs.append(val)
			f.close()
			f = open(name + '_magical', "w")
			for c in coefs:
				f.write("%.40f\n" % c)
			f.close()
		coefs = []


#Выполняем случайные мутации коэффициентов нейросети
#Мутация сохраняется, только в случае увеличения произведения очков на обоих уровнях
#Иногда вызывалась вручнуб во время обучения
def random():
	global nw
	nw = neuronweb.neuronweb(n_layers, [n_hidden, n_actions], n_features)
	nw.read_coefs();
	best_s = run_level("levels/train_level.data") * run_level("levels/test_level.data")
	i = 0
	while(True):
		i = i + 1
		sim_salabim()
		nw.read_coefs("_magical")
		train = run_level("levels/train_level.data")
		test = run_level("levels/test_level.data")
		cur_s = train * test
		if cur_s > best_s and train > 0:
			best_s = cur_s
			nw.write_coefs()
			sys.stdout.write(" [%d]Train: %f   Test: %f\n" % (i, train, test))
			sys.stdout.flush()
		else:
			sys.stdout.write(" [%d]Train: %f   Test: %f\r" % (i, train, test))
			sys.stdout.flush()
	nw.destroy()




#Число повторений одного и того же действия, для определения наилучшего хода
cdef int repeat 322

cdef int train(float* state):
	global last_score, prev_act
	cdef:
		float best_val = -1e9
		float* scores = <float*> calloc (4, sizeof (float))
		int has_next = 1

    #спрашиваем нейросеть
	cdef float* ans = nw.answer(state)

    #Выполняем каждое действие по 322 раза, загружаемся в текущее состояние и сохраняем набранные очки
	checkpoint = bbox.create_checkpoint()
	for i in range(4):
		for j in range(repeat):
			bbox.c_do_action(i)
		scores[i] = bbox.get_score() - last_score
		bbox.load_from_checkpoint(checkpoint)

    #Формируем массив ошибок для градиентного спуска
	cdef float maxim = -100000, minim = 100000
	for i in range(4):
		if scores[i] > maxim:
			maxim = scores[i]
		if scores[i] < minim:
			minim = scores[i]
	for i in range(4):
		scores[i] = (scores[i] - minim)/(maxim - minim)

    #совершаем действие, выданное нейросетью
	for act in range(n_actions):
		if ans[act] > best_val:
			action_to_do = act
			best_val = ans[act]

	prev_act = action_to_do
	has_next = bbox.c_do_action (action_to_do)
	last_score = bbox.get_score()
	nw.gradient_down (state, scores, 0.1)
	free(scores)
	return has_next


#основной цикл обучения. Вызываем из bot.py, когда обучаем нейронную сеть
def run_train():
	global num_of_train
	print("##############################################")
	print("Train started")
	global nw
	nw = neuronweb.neuronweb(n_layers, [n_hidden, n_actions], n_features)
	nw.read_coefs()
	for i in range(1000):
		train = run_train_level("levels/train_level.data")
		#test = run_train_level("levels/test_level.data")
		#print("[" + str(num_of_train)+ "] Train:" + str(train) + " Test:" + str(test))
		print("[" + str(num_of_train)+ "] Train:" + str(train))
		num_of_train = num_of_train + 1
		nw.write_coefs()
	print("Train finished")
	nw.destroy()



#инициализируем переменные, запускаем уровни
#в обучении данная функция не использовалась
def run():
	global nw
	nw = neuronweb.neuronweb(n_layers, [n_hidden, n_actions], n_features)
	nw.read_coefs();
	train = run_level("levels/train_level.data")
	test = run_level("levels/test_level.data")
	print("Train:" + str(train) + " Test:" + str(test))
	nw.destroy()

