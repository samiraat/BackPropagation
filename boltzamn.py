import numpy as np
from numpy import array
import random
import cPickle, gzip, numpy
import matplotlib.pyplot as plt

# Load the dataset
f = gzip.open('/home/dell/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_data = train_set[0] 
train_label = train_set[1]


def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

class boltzman:

	def __init__(self,layer_visible,layer_hidden,iteration,learning_rate):
		self.layer_visible=layer_visible
		self.layer_hidden=layer_hidden
		self.iteration=iteration
		self.learning_rate=learning_rate
		self.weight_vh=np.random.randn(self.layer_visible,self.layer_hidden) 
		self.bias_hid=np.random.randn(1,self.layer_hidden)					 
		self.bias_vis=np.random.randn(1,self.layer_visible)					 
		self.matrix=np.zeros((50000,784))

	
	def feedforward(self,input):
		z=np.dot(input,self.weight_vh) 
		g=z+self.bias_hid 
		activation_h=sigmoid(g) 
		return activation_h

	def backward(self,input):
		sum=np.dot(input, self.weight_vh.T) 
		output=sum+self.bias_vis
		output=sigmoid(output)
		return output
	


	def train(self,input):


		for i in range(self.iteration):

			for j in range(train_data.shape[0]):

				instance=train_data[j] 
				instance = instance.reshape((1,784)) 

				round1=self.feedforward(instance) 
				round2=self.backward(round1) 	  
				round3=self.feedforward(round2)	  
				self.matrix[j]=round2
					
				positive=np.dot(instance.T,round1) 
				ngative=np.dot(round2.T,round3)	   

				self.weight_vh += self.learning_rate* (positive - ngative)
				self.bias_vis += self.learning_rate*(instance-round2)
				self.bias_hid +=self.learning_rate*(round1-round3)
				e=np.dot(np.dot(instance,self.weight_vh),round1.T)
				enrgy=(-1*(((np.dot(self.bias_vis,instance.T))+(np.dot(self.bias_hid,round1.T))+e )))
	
				
		for i in range(5):
			rand_instance=np.random.randint(50000, size=5)
			rand=rand_instance[i]

				
			v =self.matrix[rand].reshape((28, 28))
			plt.imshow(v, cmap='gray')
			print "label is ", train_label[rand]
			plt.show()

				

boltz=boltzman(784,200,iteration=5,learning_rate=0.1)
boltz.train(train_data)