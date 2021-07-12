import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal

x = np.arange(0,1,0.001)
y = np.cos(x)
z = np.sin(x)

plt.pcolormesh(x, y,z, vmin=0, vmax=5, shading='flat')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()