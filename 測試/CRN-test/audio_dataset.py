import tensorflow as tf
import numpy as np
import os
import random

from tensorflow._api.v2 import audio, data
from tensorflow.python.module.module import camel_to_snake

def check_class(filedir):
    '''
    檢查檔案名稱，當前目錄下必須包含 "clean_train" 和 "noise_train" 兩個目錄
    '''
    print("In: ", filedir)
    print("Check if 'clean_train' and 'noise_train' exist")
    check_list = [f for f in os.listdir(filedir)]
    
    if('clean_train' in check_list and 'noise_train' in check_list):
        return True
    else:
        return False

def path_to_audio(filedir):
    data = tf.io.read_file(filedir)
    data, sr = tf.audio.decode_wav(data)
    return data, sr


def mix_audio_with_noise(clean, noise):
    if(len(clean) < len(noise)):
        noise = noise[:len(clean)]
    else:# 面對當前 dataset 不會有問題而已
        noise = np.append(noise, noise[:(len(clean)-len(noise))])
    
    scaler = np.abs(np.random.normal(0.6,0.2,1))
    return clean + noise * scaler


def audio_dataset_generator(CLEAN:tf.data.Dataset, NOISE:tf.data.Dataset):
    # 將噪音放大縮小的常數
    while True:
        clean_wav, sr = next(CLEAN.take(1).as_numpy_iterator())
        noise_wav, sr = next(NOISE.take(1).as_numpy_iterator())

        #clean_wav = tf.reshape(clean_wav, [len(clean_wav)])
        #noise_wav = tf.reshape(noise_wav, [len(noise_wav)])

        print("Reshape")

        mix_wav = mix_audio_with_noise(clean_wav,noise_wav)
        yield mix_wav, clean_wav



def get_train_dataset(filedir):
    # check file
    assert check_class(filedir=filedir), "檔案目錄名稱沒有 clean_train 或 noise_train"

    # create two dataset, clean and noise
    CLEAN = tf.data.Dataset.list_files(str(filedir+os.sep+"clean_train"+os.sep+"*")).shuffle(64)
    NOISE = tf.data.Dataset.list_files(str(filedir+os.sep+"noise_train"+os.sep+"*")).shuffle(64)
    
    CLEAN = CLEAN.map(path_to_audio)
    NOISE = NOISE.map(path_to_audio)

    CLEAN = CLEAN.map(lambda data,sr:(tf.reshape(data,[len(data)]),sr))
    NOISE = NOISE.map(lambda data,sr:(tf.reshape(data,[len(data)]),sr))

    MIX_AUDIO = tf.data.Dataset.from_generator(
        lambda: audio_dataset_generator(CLEAN, NOISE),
        output_types=(tf.float32,tf.float32),
        output_shapes=((None,),(None,))
    )

    return MIX_AUDIO
    