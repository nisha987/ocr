class Network(object):

	def _init_(self, sizes)
	self.mun_layers = len(sizes)
	self.sizes=sizes
	self.biases=[np.random.randn(y,1) for y in sizes[1:]]
	self.weights=[np.random.randn(y,x) for x,y zip(sizes[:-1], sizes[1:])]


def sigmoid(z)
	return 1.0/(1.0 + np.exp(-z)) 			# z is numpy array


def feedforward(self ,a):
	for b, w in zip(self.biases, self.weights):
		a= sigmoid(np.dot(w, a)+b)		# it applies sigmoid function to each element in array
	return a


def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
	if test_data: n_test = len(test_data)
	n=len(training_data)
	for j in xrange(epochs):
		random.shuffle(training_data)
		mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
		for mini_batche in mini_batches:
			self.update_mini_batch(mini_batch, eta)		#in each mini_batch we apply SGD
		if test_data:
			print "Epoch {0}: {1}/ {2}".format(j,self.evaluate(test_data), n_test)
		else
			print "Epoch {0} complete".format(j)


def update_mini_batch(self, mini_batch, eta):
	nabla_b = [np.zeros(b.shape) for b in self.biases]
	nabla_w = [np.zeros(w.shape) for w in self.weights]
	for x, y in mini_batch:
		delta_nabla_b, delta_nabla_w = self.backdrop(x, y)
		nabla_b=[nb+ dnb for dnb in zip(nabla_b, delta_nabla_b)]
		nabla_w=[nw+ dnw for dnw in zip(nabla_w, delta_nabla_w)]
	self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
	self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
		







