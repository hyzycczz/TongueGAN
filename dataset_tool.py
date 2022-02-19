
from cProfile import label
from email.mime import image
import os
import sys
import shutil
import argparse
import cv2
from cv2 import IMWRITE_PAM_FORMAT_GRAYSCALE
import numpy as np
# ------------------- arg -------------------
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ton', help="dataset 的路徑")
parser.add_argument('--output', type=str, default='inputData', help="轉換後要放的路徑")

args = parser.parse_args()

# ------------------- util -------------------
def pre_config(dataPath, outputPath):
    '''
    檢查 dataPath 與 outputPath 路徑：
    '''
    try:
        if(not os.path.isdir(dataPath)):
            raise OSError("dataPath:{} 不存在".format(dataPath))
        
        if(not os.path.isdir(outputPath)):
            os.mkdir(outputPath)
            os.mkdir(os.path.join(outputPath, "label"))
            os.mkdir(os.path.join(outputPath, "image"))
        else:
            x = input("outputPath:{} 存在，是否要清空路徑 (y/n):".format(outputPath)).lower()
            if(x == 'y'):
                shutil.rmtree(os.path.join(outputPath))
                os.mkdir(outputPath)
                os.mkdir(os.path.join(outputPath, "label"))
                os.mkdir(os.path.join(outputPath, "image"))
            else:
                raise OSError("一定要清空現有路徑，請重新啟動程式")

    except OSError as osErr:
        print("OSError: {}".format(osErr))
        sys.exit()

    return


def recursiveFindFile(dataPath, outputPath):
    '''
    將路徑中的資料都複製到 outputPath：
        - 使用 DFS 的方式列出 dataPath 目錄底下的所有接受檔案，並複製到 outputPath 目錄。
            接受檔案為：txt, jpg, png, JPG, PNG
    '''
    print("資料複製中：")

    acceptedFileFormat = ["txt", "jpg", "png", "JPG", "PNG"]
    DFSlist = [os.path.join(dataPath,file) for file in os.listdir(dataPath)] # with relative directory

    while(DFSlist):
        file = DFSlist.pop()
        if(file[-3:] in acceptedFileFormat):
            print("moveing {} to {}".format(file.split(os.sep)[-1], outputPath))
            if(file[-3:] == "txt"):
                shutil.copy2(file, os.path.join(outputPath, "label"))
            else:
                shutil.copy2(file, os.path.join(outputPath, "image"))    
            continue

        if(os.path.isdir(file)):
            curList = [os.path.join(file, newfile) for newfile in os.listdir(file)]
            DFSlist.extend(curList)

    return

def cropingImage(outputPath):
    '''
    依據 label 切割資料：
        - 先列出所有 labels 和 images 檔名，
        - 再輸入 cropping 函式時，將完整路徑 join 起來。
    '''
    print("切割中:")
    Path = os.path.join(outputPath, "cropped")
    labelPath = os.path.join(outputPath, "label")
    imagePath = os.path.join(outputPath, "image")
    #os.mkdir(Path)
    
    real_labels = os.listdir(labelPath)
    images = os.listdir(imagePath)
    
    fake_labels = list(map(lambda x: x[:-3] +'txt', images))

    zipdata = list(zip(fake_labels, images))
    
    for labelName, imageName in zipdata:
        assert labelName[:-4] == imageName[:-4], "{} 和 {}檔名不同".format(labelName, imageName)
        img = cropping(os.path.join(imagePath, imageName) ,os.path.join(labelPath, labelName))
        if(img is not None):
            cv2.imwrite(os.path.join(Path, "{}.png".format(labelName[:-4])), img)
    
    return


def cropping(imagedir, labeldir):
    img = cv2.imread(imagedir)
    try:
        with open(labeldir) as f:
            label = f.read().split()
    except:
        print("missing: {}".format(labeldir))
        return
    
    print("cropping: {}".format(imagedir))
    label = [float(i) for i in label]
    H, W, C = img.shape

    w = int(label[3] * W)
    h = int(label[4] * H)
    y = int(label[1] * W - w/2)
    x = int(label[2] * H - h/2)

    return img[x:x+h, y:y+w, :]
    


# ------------------- main -------------------
if __name__ == "__main__":
    DATAPATH = args.dataset
    OUTPATH = args.output

    pre_config(DATAPATH, OUTPATH) #確認 DATAPATH 與建立 OUTPATH

    recursiveFindFile(DATAPATH, OUTPATH) #建立新的資料至 OUTPATH

    cropingImage(OUTPATH) #依照 label 切割資料


    