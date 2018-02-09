import matplotlib.pyplot as plt

# Draw UCI training accuracy graph

ideal = [0.9042, 0.9238, 0.9283, 0.9342, 0.9341, 0.941, 0.9415, 0.9439, 0.9421, 0.9416, 0.9422, 0.9447, 0.9458, 0.9469, 0.9462, 0.9461, 0.9467, 0.9486, 0.9458, 0.9468, 0.9472, 0.948, 0.9477, 0.948, 0.9484, 0.9479, 0.9477, 0.9489, 0.949, 0.9477, 0.9497, 0.9489, 0.9499, 0.9493, 0.9492, 0.949, 0.9494, 0.9498, 0.9487, 0.9497] 

real = [0.9039, 0.9242, 0.9272, 0.9346, 0.934, 0.9412, 0.9412, 0.943, 0.9412, 0.9413, 0.9431, 0.9443, 0.9459, 0.9463, 0.9461, 0.9467, 0.9468, 0.9484, 0.946, 0.9466, 0.9469, 0.9487, 0.9474, 0.949, 0.9483, 0.9466, 0.9477, 0.9487, 0.949, 0.9477, 0.9495, 0.9496, 0.9499, 0.9495, 0.9489, 0.9486, 0.9489, 0.9493, 0.9485, 0.9493]

real_0_1 = [0.9003, 0.9099, 0.9134, 0.9173, 0.9176, 0.9246, 0.9269, 0.931, 0.9261, 0.9141, 0.9303, 0.9234, 0.929, 0.9334, 0.9344, 0.9323, 0.9383, 0.925, 0.9347, 0.9173, 0.9299, 0.9355, 0.93, 0.9329, 0.9363, 0.9373, 0.9304, 0.9298, 0.9293, 0.9329, 0.9401, 0.9285, 0.9352, 0.9316, 0.9308, 0.9377, 0.9379, 0.9373, 0.9374, 0.9417]
real_0_2 = [0.8718, 0.8789, 0.8664, 0.8887, 0.8391, 0.8831, 0.8689, 0.874, 0.8325, 0.889, 0.8741, 0.8669, 0.8785, 0.8912, 0.8628, 0.8687, 0.8784, 0.863, 0.8739, 0.8736, 0.889, 0.8573, 0.8856, 0.891, 0.852, 0.8962, 0.8942, 0.886, 0.8896, 0.8116, 0.8094, 0.8886, 0.9139, 0.8808, 0.8905, 0.8833, 0.8727, 0.8741, 0.8802, 0.8825]
 
real_0_3 = [0.8009, 0.835, 0.8342, 0.8072, 0.8183, 0.7567, 0.7797, 0.8382, 0.817, 0.8062, 0.8354, 0.7522, 0.7408, 0.7623, 0.7494, 0.8112, 0.7332, 0.8156, 0.8002, 0.7825, 0.7403, 0.7546, 0.8432, 0.7411, 0.7528, 0.8224, 0.8188, 0.7643, 0.7367, 0.7366, 0.7413, 0.7455, 0.7906, 0.7075, 0.792, 0.7395, 0.7901, 0.6982, 0.7513, 0.7517] 
real_0_4 = [0.6632, 0.7604, 0.694, 0.5988, 0.6377, 0.7291, 0.7044, 0.7054, 0.6312, 0.6479, 0.5902, 0.6384, 0.674, 0.6502, 0.5822, 0.6253, 0.642, 0.6364, 0.6858, 0.615, 0.6891, 0.6431, 0.656, 0.632, 0.7156, 0.5947, 0.6878, 0.687, 0.6843, 0.5857, 0.6842, 0.6768, 0.6653, 0.6318, 0.577, 0.6371, 0.6155, 0.6339, 0.6913, 0.6286]
real_0_5 = [0.6472, 0.6704, 0.6222, 0.5908, 0.6305, 0.5448, 0.6161, 0.5706, 0.5633, 0.6252, 0.6202, 0.6014, 0.5115, 0.5912, 0.4312, 0.4752, 0.5011, 0.4577, 0.476, 0.6085, 0.4734, 0.5317, 0.4538, 0.5454, 0.4952, 0.4853, 0.5623, 0.5672, 0.5191, 0.4964, 0.5487, 0.6125, 0.5486, 0.5727, 0.5803, 0.4677, 0.4454, 0.4278, 0.587, 0.4758]

# line 1 points
x = range(0,40)

#plt.yscale('log')
plt.ylim(.3, 1.0)
plt.grid(True)
# plotting the line 1 points 
plt.plot(x, ideal, label = "ideal")
 
plt.plot(x, real, label = "real")
plt.plot(x, real_0_1, label = "s.d.=0.1")
plt.plot(x, real_0_2, label = "s.d.=0.2")
plt.plot(x, real_0_3, label = "s.d.=0.3")
plt.plot(x, real_0_4, label = "s.d.=0.4")
plt.plot(x, real_0_5, label = "s.d.=0.5")
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


