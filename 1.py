# Testing for nparray drawing and dumping
import numpy as np


#a = np.arange(1600).reshape(20,80)
#print(a)


#np.save('test3.npy', a)  


# parse input distribution data
# return distribution, max, min
distribution = []
distribution_path = "./distribution.txt"
distribution_max, distribution_min = None, None
distribution_stage = None
def read_distribution(path):
    rtn = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            tmp = line.split()
            rtn.append([float(tmp[1]), float(tmp[2])])
    return rtn, rtn[-1][0], rtn[0][0]

distribution, distribution_max, distribution_min = read_distribution(distribution_path)
distribution_stage = len(distribution)

print distribution, distribution_max, distribution_min
print "------------------"
# find distribution parameter 
# conductance -1 to 1 value,distribution array, max, min of conductance, total stages
# return mean, stdv
def check_distribution(c, d, hi, lo, stage):
    idx = int((c+1.0)*stage/2)
    if idx >= stage-1: 
        m, var = d[-1][0], d[-1][1]
    else:
        m, var = (d[idx][0]+d[idx+1][0])/2.0, (d[idx][1]+d[idx+1][1])/2.0
    return (m-lo)/(hi-lo)*2.0 - 1.0, var

u, var = check_distribution(-0.1, distribution, distribution_max, distribution_min, distribution_stage)

#print u, var

# normal sample with mean and var
def new_weight(mean, var):
    return var * np.random.randn()+mean

print new_weight(u,var)
