from asyncore import file_dispatcher
import os
import cv2


PATH = 'inputData/image'

filedir = [os.path.join(PATH, file) for file in os.listdir(PATH)]


for file in filedir:
    print('processing {file}'.format(file=file))
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[:,:,3] = [[0 if sum(img[j][i][:3])==0 else 255 for i in range(len(img[0]))] for j in range(len(img))]
    cv2.imwrite(file, img)