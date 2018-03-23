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
#print("X:::",X,"\n","Y:::",Y)
#print(X.max())

trainx = []
#trainy = []
testx = []
#testy = []

idx = len(trainX)
trainX = X[:idx]
trainy = Y[:idx]
testX = X[idx:]
testy = Y[idx:]

#trainY_onehot = []
#for i in Y:
#    trainY_onehot.append([0]*20)
#    trainY_onehot[-1][int(i)] = 1
#Y = np.asarray(trainY_onehot)


print(testX.shape, testY.shape, trainX.shape, trainY.shape)


trainX = np.tile(trainX, (80,1))
trainY = np.tile(trainY, (80,1))
testX = np.tile(testX, (80,1))
testY = np.tile(testY, (80,1))

for i in range(len(trainX)):
    trainX[i] = trainX[i]+ np.random.normal(0,0.01,len(trainX[0]))

for i in range(len(testX)):
    testX[i] = testX[i]+ np.random.normal(0,0.01,len(testX[0]))

# idx shuffle
#idx = np.random.randint(dataX.shape[0], size = dataX.shape[0]/5)
#for i in rage(dataX.shape[0]):
#    if i in idx:
#        testX.append(dataX[i])
#        testY.append(labelY[i])
#    else:
#        trainX.append(dataX[i])
#        trainY.append(labelY[i])

#idx = dataX.shape[0]/5


testY = np.asarray(testY)
testX = np.asarray(testX)#[:,::4])
trainX = np.asarray(trainX)#[:,::4])
trainY = np.asarray(trainY)

print(testX.shape, testY.shape, trainX.shape, trainY.shape)
print(trainY, testY)

np.save('trainX', trainX)
np.save('trainY', trainY)
np.save('testX', testX)
np.save('testY', testY)
