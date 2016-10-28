import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
import  matplotlib as mpl
from matplotlib.patches import Circle

training_sample_x = np.random.uniform(0, 10, 1000)
training_sample_y = np.random.uniform(0, 10, 1000)


in_round = []
for i in range(1000):
    if (training_sample_x[i] - 5) ** 2 + (training_sample_y[i] - 5) ** 2 < 16:
        in_round.append([1])
    else:
        in_round.append([0])


plt.plot(training_sample_x, training_sample_y, 'o')

plt.show()

net = nl.net.newff([[0, 10], [0, 10]], [50, 1])
err = net.train(np.column_stack((training_sample_x, training_sample_y)), in_round, epochs=1000, show=100)



check_x = np.random.uniform(0, 10, 100)
check_y = np.random.uniform(0, 10, 100)


plt.plot(check_x, check_y, 'o')
for i in range(100):
    if net.sim([[check_x[i], check_y[i]]]).round() == 1:
        plt.plot(check_x[i], check_y[i], 'o', color='red')

ellipse = mpl.patches.Circle((5, 5), 4)
fig, ax = plt.subplots()
fig.gca().add_artist(ellipse)

plt.show()



