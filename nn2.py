import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def showPlot(inputArray, size_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, 1, 1 / size_grid)
    y = np.arange(0, 1, 1 / size_grid)
    xgrid, ygrid = np.meshgrid(x, y)
    ax.plot_surface(xgrid, ygrid, inputArray, rstride=1, cstride=1)
    plt.show()

training_sample_x1 = np.random.normal(2.5, 1, 100)
training_sample_x2 = np.random.normal(5, 1, 100)
training_sample_x3 = np.random.normal(7, 1, 100)
training_sample_y1 = np.random.normal(3, 1, 100)
training_sample_y2 = np.random.normal(7, 1, 100)
training_sample_y3 = np.random.normal(3, 1, 100)

training_sample_x = np.hstack((training_sample_x1, training_sample_x2, training_sample_x3))
training_sample_y = np.hstack((training_sample_y1, training_sample_y2, training_sample_y3))

training_sample_z = []
for i in range(0, 100):
    training_sample_z.append([0])
for i in range(100, 200):
    training_sample_z.append([0.5])
for i in range(200, 300):
    training_sample_z.append([1])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(training_sample_x, training_sample_y, training_sample_z, 'o')
plt.show()


net = nl.net.newff([[0, 10], [0, 10]], [100, 1])
err = net.train(np.column_stack((training_sample_x, training_sample_y)), training_sample_z, epochs=300, show=100)


check_x1 = np.random.normal(2.5, 1, 15)
check_x2 = np.random.normal(5, 1, 15)
check_x3 = np.random.normal(7, 1, 15)
check_y1 = np.random.normal(3, 1, 15)
check_y2 = np.random.normal(7, 1, 15)
check_y3 = np.random.normal(3, 1, 15)


check_x = np.hstack((check_x1, check_x2, check_x3))
check_y = np.hstack((check_y1, check_y2, check_y3))


# plt.plot(check_x, check_y, 'o',  color='black')
for i in range(45):
    if (net.sim([[check_x[i], check_y[i]]]) * 2).round() == 0:
        plt.plot(check_x[i], check_y[i], 'o', color='red')
    if (net.sim([[check_x[i], check_y[i]]]) * 2).round() == 1:
        plt.plot(check_x[i], check_y[i], '*', color='green')
    if (net.sim([[check_x[i], check_y[i]]]) * 2).round() == 2:
        plt.plot(check_x[i], check_y[i], '+', color='blue')


plt.show()



