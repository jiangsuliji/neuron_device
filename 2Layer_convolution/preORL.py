# this preprocess the data and introduce gaussian noise to enrich the dataset

import numpy as np


data = np.load('ORL_faces.npz') #assumption is that the data is unzipped
trainX = data['trainX']
testX = data['testX']
trainY= data['trainY']
testY= data['testY']

trainY = np.eye(20)[trainY]
testY  = np.eye(20)[testY]
 
X = np.concatenate((trainX, testX), axis = 0)
Y = np.concatenate((trainY, testY), axis = 0)

# scaling
#print(X.max(),X.min(),np.mean(X))
a,b=X.max(),X.min()
X = (X-b)*2.0/(a-b)-1
print("X:::",X,"\n","Y:::",Y)
#print(X.max())

#trainY_onehot = []
#for i in Y:
#    trainY_onehot.append([0]*20)
#    trainY_onehot[-1][int(i)] = 1
#Y = np.asarray(trainY_onehot)
print(X.shape, Y.shape)

dataX = np.tile(X, (80,1))
labelY = np.tile(Y, (80,1))
#print(dataX.shape, labelY.shape)
#print(dataX)

for i in range(len(dataX)):
    dataX[i] = dataX[i]+ np.random.normal(0,0.01,len(dataX[0]))

#print(dataX)

print(dataX.shape, labelY.shape)

trainX = []
trainY = []
testX = []
testY = []
# idx shuffle
idx = np.random.randint(dataX.shape[0], size = dataX.shape[0]/5)
for i in range(dataX.shape[0]):
    if i in idx:
        testX.append(dataX[i])
        testY.append(labelY[i])
    else:
        trainX.append(dataX[i])
        trainY.append(labelY[i])

testY = np.asarray(testY)
testX = np.asarray(testX)
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)

print(testX.shape, testY.shape, trainX.shape, trainY.shape)
np.save('testX', testX)
np.save('testY', testY)
np.save('trainX', trainX)
np.save('trainY', trainY)



