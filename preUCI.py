# Preprocess UCI dataset

import numpy as np

filename = "UCI/train.txt"
data = np.genfromtxt(filename, delimiter=',')

label = data[:,64]
data = data[:,:64]

print(data.shape, label.shape)
print(data)
print(label)

np.save('training_data.npy',data)
np.save('training_label.npy',label)


filename = "UCI/test.txt"
data = np.genfromtxt(filename, delimiter=',')

label = data[:,64]
data = data[:,:64]

print(data.shape, label.shape)
print(data)
print(label)

np.save('test_data.npy',data)
np.save('test_label.npy',label)



