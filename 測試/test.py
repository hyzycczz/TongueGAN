import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal

x = np.arange(0,1,0.1)
y = np.arange(0,1,0.1)
z = [[] for i in range(len(x))]


for i in range(len(x)):
    for j in range(len(y)):
        z[i].append(x[i] + y[j])



plt.pcolormesh(x, y,z, vmin=0, vmax=2, shading='flat')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()