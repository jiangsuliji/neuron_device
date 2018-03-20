import matplotlib.pyplot as plt

# Draw UCI training accuracy graph

ideal =[0.9042, 0.9238, 0.9283, 0.9342, 0.9341, 0.941, 0.9415, 0.9439, 0.9421, 0.9416, 0.9422, 0.9447, 0.9458, 0.9469, 0.9462, 0.9461, 0.9467, 0.9486, 0.9458, 0.9468, 0.9472, 0.948, 0.9477, 0.948, 0.9484, 0.9479, 0.9477, 0.9489, 0.949, 0.9477, 0.9497, 0.9489, 0.9499, 0.9493, 0.9492, 0.949, 0.9494, 0.9498, 0.9487, 0.9497] 

real = [0.8543, 0.8881, 0.8886, 0.8886, 0.8869, 0.913, 0.9051, 0.9134, 0.9121, 0.9153, 0.9104, 0.917, 0.9078, 0.9045, 0.9142, 0.9174, 0.9119, 0.9162, 0.9108, 0.9103, 0.9136, 0.9117, 0.9155, 0.9155, 0.9024, 0.9197, 0.9164, 0.9178, 0.9172, 0.9106, 0.9182, 0.9189, 0.9172, 0.9179, 0.9194, 0.9131, 0.9189, 0.9152, 0.9125, 0.9123]


# line 1 points
x = range(0,40)

#plt.yscale('log')
plt.ylim(.8, 1.0)
plt.grid(True)
# plotting the line 1 points 
plt.plot(x, ideal, label = "ideal")
 
plt.plot(x, real, label = "real")
#plt.plot(x_ideal, y_real, label = "real")
#plt.semilogy(x_ideal, y_ideal, label="ideal")
#plt.semilogy(x_ideal, y_real, label = "real(6bit)")

# naming the x axis
plt.xlabel('Epoch')
# naming the y axis
plt.ylabel('Accuracy(%)')
# giving a title to my graph
plt.title('Large digits')

# show a legend on the plot
plt.legend(loc='lower center', ncol=4)

#plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

# function to show the plot
plt.show()


