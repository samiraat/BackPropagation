import numpy as np
from numpy import array
import random
import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('/home/dell/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
# first item of train set
#print train_set[0][0]    
# label of first item in train set
#print train_set[1][0]
#print train_set
f.close()

target=np.zeros((50000,1))
for i in range(50000):
	target[i]=train_set[1][i]

train_dat=np.zeros((50000,784))
for i in range(50000):
	train_dat[i]=train_set[0][i]


def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
def derive_sigmoid(z):
	return sigmoid(z)*(1-sigmoid(z))


class neuralnetwork:
	
	
	#number of nurons of each layer
	# h=number of hidden layer  i=number input layer hh=second hidden layer o=output
	def __init__(self,layer_i,layer_h,layer_hh,layer_o,iteration,learning_rate):
		self.lay_i=layer_i
		self.lay_h=layer_h
		self.lay_hh=layer_hh
		self.lay_o=layer_o
		self.learning_rate=learning_rate
		self.iteration=iteration

		#define bias & initialize
		self.bias_h=np.random.randn(self.lay_h,)
		self.bias_hh=np.random.randn(self.lay_hh,)
		self.bias_o=np.random.randn(self.lay_o,)
		#initiaze weight with random 
		self.weight_ih=np.random.randn(self.lay_i,self.lay_h)
		if self.lay_hh > 0 :
			self.weight_hh=np.random.randn(self.lay_h,self.lay_hh)
			self.weight_ho=np.random.randn(self.lay_hh,self.lay_o)
		else:
			self.weight_ho=np.random.randn(self.lay_h,self.lay_o)
			
		#h= activation hidden o=activation output hh=activation second hidden
		self.activation_i=np.ones((self.lay_i))
		self.activation_h = np.ones((self.lay_h))
		self.activation_o = np.ones((self.lay_o))
		self.activation_hh = np.ones((self.lay_hh))
		#changes update weughts in every layer
        
	def feedforward(self, input):
		self.activation_i=input
		z=np.dot(input.T,self.weight_ih)
		g=z+self.bias_h
		self.activation_h=sigmoid(g)

		if self.lay_hh>0 :
			z1=np.dot(self.activation_h,self.weight_hh)
			g1=z1+self.bias_hh
			self.activation_hh=sigmoid(g1)
			z2=np.dot(self.activation_hh,self.weight_ho)
			g2=z2+self.bias_o
			self.activation_o=sigmoid(g2)
		else:
			z3=np.dot(self.activation_h,self.weight_ho)
			g3=z3+self.bias_o
			self.activation_o=sigmoid(g3)
		return self.activation_o


	def backpropagation(self, target):
		error=target - self.activation_o 
		deriv=derive_sigmoid(self.activation_o)
		#if output layer 
		delta_output=error*deriv
		#change varible is for changes in every layer
		change_o = delta_output * np.reshape(self.activation_h, (self.activation_h.shape[0],1))

		if self.lay_hh >0 :
			sum1=np.dot(self.weight_ho,delta_output)
			delta_hidden2=derive_sigmoid(self.activation_hh)*sum1
			change_hh=delta_hidden2 * np.reshape(self.activation_h, (self.activation_h.shape[0],1))

			sum2=np.dot(self.weight_hh,delta_hidden2)
			delta_hidden1=derive_sigmoid(self.activation_h)*sum2
			change_h=delta_hidden1 * np.reshape(self.activation_i, (self.activation_i.shape[0],1))
			#hidden layer

		else :
			sum =np.dot(self.weight_ho,delta_output)
			delta_hidden=derive_sigmoid(self.activation_h)*sum
			change_h =delta_hidden * np.reshape(self.activation_i, (self.activation_i.shape[0], 1))
		#update weights & bias
		#if output layer
	
		self.weight_ho=self.weight_ho+(self.learning_rate*self.activation_o*delta_output)
		self.bias_o=self.bias_o+(self.learning_rate*self.activation_o*delta_output)
		#if hidden layer 2
		if self.lay_hh>0 :
			self.weight_hh=self.weight_hh+(self.learning_rate*self.activation_hh*delta_hidden2)
			self.bias_hh=self.bias_hh+(self.learning_rate*self.activation_hh*delta_hidden2)
			self.weight_ih=self.weight_ih+(self.learning_rate*self.activation_h*delta_hidden1)
			self.bias_h=self.bias_h+(self.learning_rate*self.activation_h*delta_hidden1)
		else:
			#if hidden layer for 3 layer
			self.weight_ih=self.weight_ih+(self.learning_rate*self.activation_h*delta_hidden)
			self.bias_h=self.bias_h+(self.learning_rate*self.activation_h*delta_hidden)
	
		#print error
		return (error)**2

	def train(self,train_data,target):
		for i in range(self.iteration):
			error=0.0
			random.shuffle(train_data)
			for j in range(train_data.shape[0]):
				input=train_data[j]
				targe=np.zeros(self.lay_o,)
				targe[target[i][0]]=1
				self.feedforward(input)
				error=error+self.backpropagation(targe)
			error = 0.0
			for j in range(50000):
				tr = train_data[j]
				tg = target[j][0]
				f = self.feedforward(tr)
				s = sum(f)
				error += (1.0 - f[tg]/s)**2

			error = np.sum(error) / 50000
			print 'epoch:' ,i , 'loss : ',0.5*(error)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		# difference the best with the target
		return sum(int(x == y) for (x, y) in test_results)



#bp = neuralnetwork(784,600,0, 10, iteration= 200, learning_rate = 0.1)	
bp = neuralnetwork(784,100,0, 10, iteration= 200, learning_rate = 0.1)
#bp = neuralnetwork(784,100,100, 10, iteration= 200, learning_rate = 0.1)
#bp = neuralnetwork(784,600,600, 10, iteration= 200, learning_rate = 0.1)
bp.train(train_dat,target)
mlp.evaluate(test_set)

