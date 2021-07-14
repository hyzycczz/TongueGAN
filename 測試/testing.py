import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal

rng = np.random.default_rng(seed=1234)
t , t = 10, 2
fs = 10e3
N = 1e5

amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
print("TIME: ", time)
mod = 500*np.cos(2*np.pi*0.25*time)

carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = rng.normal(scale=np.sqrt(noise_power),
                   size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise # 聲音

plt.figure()
plt.plot(time, x)
plt.xlim(0,0.1)
plt.show()

f, t, Zxx = signal.stft(x, fs, nperseg=1000)

plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='flat')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
