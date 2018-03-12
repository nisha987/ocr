class Network(object):

	def _init_(self, sizes)
	self.mun_layers = len(sizes)
	self.sizes=sizes
	self.biases=[np.random.randn(y,1) for y in sizes[1:]]
	self.weights=[np.random.randn(y,x) for x,y zip(sizes[:-1], sizes[1:])]

def sigmoid(z)
	return 1.0/(1.0 + np.exp(-z))

