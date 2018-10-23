import numpy as np
import glob,cv2,argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
parser = argparse.ArgumentParser(description='This is an similarity checker')
parser.add_argument('-f', dest='folder', type=str, default="/home/schwarz/Pictures/topit/", help='select a folder')
args = parser.parse_args()
folder = args.folder

def describe_(imagePath):
    image = cv2.cvtColor(cv2.imread(imagePath),cv2.COLOR_BGR2HSV)
    features = []
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2
    segments = ((0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h))
    ellipMask = np.zeros((h,w), dtype=np.uint8)
    cv2.ellipse(ellipMask, (cX, cY), (w*3//8,h*3//8), 0, 0, 360, 255, -1)
    for (startX, endX, startY,endY) in segments:
        cornerMask = np.zeros((h,w), dtype=np.uint8)
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)
        features.extend(histogram(image, cornerMask))
    features.extend(histogram(image, ellipMask))
    return  imagePath.replace(folder,''),features

def histogram(image, mask):
    hist = cv2.calcHist([image],[0,1,2],mask,(8, 12, 3),
            [0,180,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def describe(imagePath):
    image = cv2.cvtColor(cv2.imread(imagePath),cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2
    ellipMask = np.zeros((h,w), dtype=np.uint8)
    cv2.ellipse(ellipMask, (cX, cY), (cX, cY), 0, 0, 360, 255, -1)
    hist = cv2.calcHist([image],[0,1,2],ellipMask,(8, 12, 3),
            [0,180,0,256,0,256])
    return imagePath.replace(folder,''),\
            cv2.normalize(hist, hist).flatten()

def chi2_distance(row):
    score = np.sum(np.square((row[0][1]-row[1][1]))/(row[0][1]+row[1][1]+(1e-10)))
    if score <= threshold:
        print('[*] found %s from %s, the score is %f' % (row[1][0], row[0][0],score))
        return row[1][0], row[0][0]
    else: return 0,0

threshold = 0.1
pool = ProcessPoolExecutor(max_workers=4)
index = list(pool.map(describe_, glob.glob(folder+"*.jpg")))
print("\n[+] the index file has been generated..")
print("[+] now pairing images..")
pool.map(chi2_distance, list(combinations(index,2)))
