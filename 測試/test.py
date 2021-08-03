from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift

import matplotlib.pyplot as plt
import numpy as np

samplerate, data = wavfile.read('inputaudio.wav')
# samplerate 是取樣率
# data 是音訊

duration = len(data)/samplerate
# 利用資料的長度除以取樣率會是整個音訊的長度，單位 秒。

time = np.arange(0,duration,1/samplerate)
# 用 np 建立個 0 ~ duration 時間的資料，資料之間的間隔為 1/samplerate

print(samplerate)

plt.figure()

plt.subplot(2,1,1)
plt.plot(time,data)

fftdata = fft(data)
fftx = fftfreq(data, 1/samplerate)[:data//2]

plt.subplot(2,1,2)
plt.plot(fftdata, 2.0/data)

plt.show()


from scipy.fft import fft, fftfreq
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()