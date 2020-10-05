import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("P1_LIT101.csv", delimiter=',', skiprows=1, dtype='str')
data = data[:, 1:]
data = data.astype(float)

data[data[:, 1]==0, 1] = 1

u1 = data[:, 1]
u2 = data[:, 2]
u1 = data[:, 1]-1
u2 = data[:, 2]-1

k1 = np.logical_and(u1 == 1, u2 == 0)



ax1 = plt.gca()
ax2 = ax1.twinx()
tmp = data[k1]
ax1.plot(data[:, 0])
ax1.plot(tmp[:100, 0])
ax2.plot(u1, alpha=0.8, c='tab:red')
ax2.plot(u2, alpha=0.8, c='tab:green')

ax1.set_ylim([0, 1000])
ax2.set_ylim([-1, 5])
plt.grid()
plt.show()