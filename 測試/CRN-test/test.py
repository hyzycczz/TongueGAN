import os
from re import L
import winsound
import numpy as np
from numpy.testing._private.utils import print_assert_equal
import tensorflow as tf
from tensorflow._api.v2 import audio
import soundfile as sf

import audio_dataset

# winsound.PlaySound(file_1, winsound.SND_FILENAME)

'''
def read_audio(filedir):
    data = tf.io.read_file(filedir)
    return tf.audio.decode_wav(data)

def get_audio_length(dir):
    data, sr = read_audio(dir)
    return len(data)/sr

def write_audio(data, sr):
    sf.write("mytune_noise.wav", data,sr)

current_dir = os.path.dirname(__file__)
dir = current_dir+"/dataset/noise_train/AirportAnnouncements_10.wav"

# dataset = tf.data.Dataset.list_files(str(dir+"/*"))
data, sr = read_audio(dir)
write_audio(data*-1, sr)
print("done")
'''

'''
min = 10000
filename = None
print(next(dataset.as_numpy_iterator()))
for file in dataset.as_numpy_iterator():
    tmp = get_audio_length(file)
    if(min > tmp):
        min = tmp
        print(min)
        filename = file

print(min)
print(filename)
print(type(dataset))
'''

''''''

current_dir = os.path.dirname(__file__)
print("目前位置: ",current_dir)
datafile = current_dir + os.sep + "dataset"

data = audio_dataset.get_train_dataset(datafile)

def double_stft(signal_A, signal_B):
    A = tf.signal.stft(signal_A,frame_length=320 ,fft_length=320,frame_step=80)
    B = tf.signal.stft(signal_B,frame_length=320 ,fft_length=320,frame_step=80)
    return A, B


data = data.map(double_stft)

for d, s in data.padded_batch(1).take(1):
    print(d.shape, type(d))
    print(s.shape)
    print()