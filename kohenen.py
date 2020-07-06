import numpy as np
from numpy import array
import random
import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('/home/dell/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
# first item of train set
#print train_set[0][0]    
#print train_set[0][1]
# label of first item in train set
#print train_set[1][0]
#print train_set[1][1]
#print train_set
target=np.zeros((50000,1))
for i in range(50000):
	target[i]=train_set[1][i]

train_data=np.zeros((50000,784))
for i in range(50000):
	train_data[i]=train_set[0][i]

f.close()
size=784
def Gaussian(x,y):
	theta=1
	z=x-y
	di=abs(z)
	return np.exp(-di/(2*(theta**2)))
def Gaussian2(x,y,x1,y1):
	theta=1
	sub_x=x-x1
	sub_y=y-y1
	sub_x=sub_x**2
	sub_y=sub_y**2
	sum=sub_x+sub_y
	di=np.sqrt(sum)
	return np.exp(-di/(2*(theta**2)))

class SOM:
	def __init__(self,number_neuron_output,iteration,learning_rate,size):
		self.neuron=number_neuron_output
		self.iteration=iteration
		self.learning_rate=learning_rate
		self.size=size
	
		#initialize weight
		self.weight_1d=np.random.randn(self.neuron,self.size)

	def train(self,input,label):
		#competition
		classeachneuron2=np.zeros((10,self.neuron))
		
		for i in range (self.iteration):
			for k in range(input.shape[0]):
				subtract=np.zeros((input.shape[1],))
				summ=np.zeros((self.neuron,))

				for j in range(self.neuron):
					subtract=np.subtract(input[k],self.weight_1d[j])
					subtract=subtract**2
					summ[j]=np.sum(subtract)
				#winner
				index_min=np.argmin(summ)
			
				classeachneuron2[int (label[k])][index_min]=classeachneuron2[int(label[k])][index_min]+1
				#cooperation
				#weight adaptation
				for j in range(self.neuron):
					g=Gaussian(index_min,j)
					self.weight_1d[j]=self.weight_1d[j]+self.learning_rate*g*(input[k]-self.weight_1d[j])
		#print classeachneuron2
		max_per_col=classeachneuron2.argmax(axis=0)
		print max_per_col
	
class SOM2d:
	def __init__(self,number_neuron_output1,number_neuron_output2,iteration,learning_rate,size):
		self.neuron1=number_neuron_output1		
		self.neuron2=number_neuron_output2
		self.iteration=iteration
		self.learning_rate=learning_rate
		self.size=size

		#initialize weight
		self.weight_2d=np.random.randn(self.neuron1,self.neuron2,self.size)
#competition
	def train(self,input,lable):
		classeachneuron=np.zeros((10,self.neuron1,self.neuron2))
	
		for itr in range(self.iteration):
			for k in range(input.shape[0]):
				subtract=np.zeros((input.shape[1],))
				summ=np.zeros((self.neuron1,self.neuron2))
				for i in range(self.neuron1):
					for j in range(self.neuron2):
						subtract=np.subtract(input[k],self.weight_2d[i][j])
						subtract=subtract**2
						summ[i][j]=np.sum(subtract)
				index_min_x=np.unravel_index(summ.argmin(),summ.shape)[0]
				index_min_y=np.unravel_index(summ.argmin(),summ.shape)[1]
				
				classeachneuron[int(lable[k])][index_min_x][index_min_y]=classeachneuron[int(lable[k])][index_min_x][index_min_y]+1
		
			#cooperation
			#weight adaptation
				for i in range(self.neuron1):
					for j in range(self.neuron2):
						g=Gaussian2(index_min_x,index_min_y,i,j)
						self.weight_2d[i][j]=self.weight_2d[i][j]+self.learning_rate*g*(input[k]-self.weight_2d[i][j])

		max_per_col2=classeachneuron.argmax(axis=0)	
		print max_per_col2

						
						




#kohhonen=SOM(10,5,0.1,784)
#kohhonen=SOM(100,5,0.1,784)
#kohhonen=SOM2d(5,5,5,0.1,784)
#kohhonen=SOM2d(10,10,5,0.1,784)
kohhonen=SOM2d(20,20,5,0.1,784)
kohhonen.train(train_data,target)