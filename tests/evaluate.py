import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_ground_truth(data, dimensions, side_lengths):
    ''' 
    data: data points
    dimensions (x=1, y=2, z=3): [dim1, dim2]
    side_lengths: [side1 length, side2 length]
    ASSUMES CONSTANT VELOCITY
    '''

    n = len(data)

    rat = side_lengths[0]/(side_lengths[0]*2+side_lengths[1]*2)
    side1_n = int(n*rat)
    side2_n = int((n-(side1_n*2)) /2)

    side1 = np.linspace(data[0, dimensions[0]], data[0, dimensions[0]]+side_lengths[0], side1_n)
    side2 = np.linspace(data[0, dimensions[1]], data[0, dimensions[1]]+side_lengths[1], side2_n)
    side3 = np.linspace(side1[-1], data[0, dimensions[0]], side1_n)
    side4 = np.linspace(side2[-1], data[0, dimensions[1]], side2_n)

    gt = np.zeros(data.shape)

    gt[:side1_n, dimensions[0]] = side1
    gt[side1_n:side1_n+side2_n, dimensions[0]] = np.ones([1, side2_n])*(side_lengths[0]+data[0, dimensions[0]])
    gt[side1_n+side2_n:(len(data)-side2_n), dimensions[0]] = side3
    gt[(len(data)-side2_n):len(data), dimensions[0]] = np.ones([1, side2_n])*data[0, dimensions[0]]

    gt[:side1_n, dimensions[1]] = np.ones([1, side1_n])*data[0, dimensions[1]]
    gt[side1_n:side2_n+side1_n, dimensions[1]] = side2
    gt[side2_n+side1_n:(len(data)-side2_n), dimensions[1]] = np.ones([1, side1_n])*(side_lengths[1]+data[0, dimensions[1]])
    gt[(len(data)-side2_n):len(data), dimensions[1]] = side4

    return gt


def path_length(data):
    
    length = 0
    data = np.array(data)

    prev_point = data[0,:]
    for point in data[0:,:]:
        dist = np.linalg.norm(point-prev_point)
        length += dist
        prev_point = point

    return length

def total_completion_time(data, time):
    return path_length(data) / time

def rmse(data, truth):
    sumdiff = 0
    for i in range(len(data)):
        diffsq = (np.subtract(truth[i], data[i]))**2
        sumdiff += diffsq
    return np.sqrt(sumdiff/len(data))

def rmse_avg(data, truth):
    return np.mean(rmse(data,truth))

if __name__ == "__main__":

    df1 = pd.read_csv(r"C:\Users\finnm\fyp\ukf3.csv")
    df2 = pd.read_csv(r"C:\Users\finnm\fyp\ekf3.csv")
    df3 = pd.read_csv(r"C:\Users\finnm\fyp\kf3.csv")
    data = np.array(df1)
    data2 = np.array(df2)
    data3 = np.array(df3)

    n = 20
    n1 = 72

    data = data[n:n1,:]
    data2 = data2[n:n1,:]
    data3 = data3[n:n1,:]

    gt = generate_ground_truth(data[:, [1,3]], [0, 1], [1.8, 1.7])

    print("kf ",rmse(data3[:, [1,3]], gt), rmse_avg(data3[:, [1,2]], gt))
    print("ekf ",rmse(data2[:, [1,3]], gt), rmse_avg(data2[:, [1,2]], gt))
    print("ukf ",rmse(data[:, [1,3]], gt), rmse_avg(data[:, [1,2]], gt))
    print(abs(path_length(data3[:, [1,3]])-path_length(gt[:, [0,1]])))
    print(abs(path_length(data2[:, [1,3]])-path_length(gt[:, [0,1]])))
    print(abs(path_length(data[:, [1,3]])-path_length(gt[:, [0,1]])))


    #plt.plot(gt[:, 0], gt[:, 1], label="Ground truth", linewidth=14, alpha=0.3, color='r')

    plt.plot(data[:,1], data[:,2], label="UKF result", color='g')
    plt.plot(data2[:,1], data2[:,2], label="EKF result", color='b')
    plt.plot(data3[:,1], data3[:,2], label="KF result", color='r')
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=2, fancybox=False, shadow=False, prop={'size': 8})
    #plt.xticks(np.arange(-0.6, 2.6, 0.2))
    #plt.yticks(np.arange(-1.2, 1.8, 0.2))
    #plt.xlim([-0.75, 1.5])
    #plt.ylim([-1, 1.25])
    plt.xlabel("y")
    plt.ylabel("z")
    plt.show()