import json
import random
import sys

import numpy as np

class QuadraticCost(object):

	@staticmethod
	def fn(a,y):
		return 0.5*np.linalg.norm(a-y)**2 

	@staticmethod
	def delta(z, a,y):
		return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(Object):

	@staticmethod
	def fn(a,y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@stativmethod
	def delta(z, a, y):
		return (a-y)

class Network(object):
	
	def __init__(self, sizes, cost==CrossEntropyCost):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.default_weight_initializer()
		self.cost=cost

	def default_weight_initializer(self):
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:1], self.sizes[1:])]
		


	def large_weight_initializer(self):
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		self.weights =[np.random.randn(y, x) for x , y in zip(self.sizes[:1], self.sizes[1:])]

	def feedforward(self,a)
		for b, w in zip(self.biases. self.weights):
			a = sigmoid(np.dot(w,a)+b)
			return a

	def SGD(self, training_data, epochs, mini_batch_size, eta. lmbda=0.0,evaluation_data= None, monitor_evaluation_cost = False, monitor_evaluation_accuracy = False, monitor_training_cost =False, monitor_training_accuracy=False):

		if evaluation_data: n_data= len(evaluation_data)
		n=len(training_data)
		evaluation_cost, evaluation_accuracy=[], []
		training_cost, training_accuracy =[], []
		
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k: k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

		for mini_batch in mini_batches: self.update_mini_batches(mini_batch, eta, lmbda, len(training_data))
			print "Epoch %s training complete" %j
			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print "Cost on training data: {} / {}". format(cost)
			
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert= True)
				training_accuracy.append(accuracy)
				print "Accuracy on training data: {} / {}".format(accuracy, n)
			
			if monitor_evaluation_cost:
				cost= self.total_cost(evaluation_data, lmbda, convert= True)
				evaluation_cost.append(cost)
				print "Cost on evaluation data: {}".format(cost)

			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print "Accuracy on evaluation data {} / {}".format(self.accuarcy(evaluation_data), n_data)
			print
			return evaluation_cost, evaluation_accuracy, \
				training_cost, training_accuracy



	def backprop(self, x, y):
		
		nabla_w = [np.zeros(b.shape) for b in self.biases]
		nabla_b = [np.zeros(w.shape) for w in self.weights]
		
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z= np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = (self.cost).delta(zs[-1], activations[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in xrange(2, self.num_layers):
			z=zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l]=delta
			nabla_w[-l]= np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	 def accuracy(self, data, convert=False):
        
        	if convert:
         		results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
       	 	else:
    	  		results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        	return sum(int(x == y) for (x, y) in results)


	def total_cost(self, data, lmbda, convert=False):
        
        	cost = 0.0
        	for x, y in data:
            		a = self.feedforward(x)
            		if convert: y = vectorized_result(y)
           		cost += self.cost.fn(a, y)/len(data)
        	cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        	return cost















