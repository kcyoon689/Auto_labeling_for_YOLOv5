# This program is for auto labeling training data of yolo v5

import os
import sys
import signal
import numpy as np
import cv2

def signal_handler(signal,frame):
	print('pressed ctrl + c!!!')
	sys.exit(0)
signal.signal(signal.SIGINT,signal_handler)

class AutoLabeler:
  def __init__(self):
    print("AutoLabeler init!")
    self.currentDirPath = os.getcwd()
    self.rawDataDirPath = self.currentDirPath + "/raw_data"
    self.rawDataFilePath_list = os.listdir(self.rawDataDirPath)
    self.rawDataFileFullPath_list = [self.rawDataDirPath + '/' + file_name for file_name in self.rawDataFilePath_list]

  def showImage(self, img):
    cv2.imshow('img', img)
    cv2.waitKey(0)

  def calcBBox(self, img):
    imgObject = np.where(img < 200) # TODO: need to tune for stable functionality
    print(imgObject)
    print(imgObject[1])
    BBoxXMin = np.min(imgObject[0]) # TODO: Check if this value is BBox X min or not
    BBoxXMax = np.max(imgObject[0])
    BBoxYMin = np.min(imgObject[1])
    BBoxYMax = np.max(imgObject[1])

    print(BBoxXMin)
    print(BBoxXMax)
    print(BBoxYMin)
    print(BBoxYMax)

    imgObjectBBox = cv2.rectangle(img, (BBoxYMin, BBoxXMin), (BBoxYMax, BBoxXMax), 100, 1) # TODO: Check cv2.rectangle get the corner position as [X,Y] order
    self.showImage(imgObjectBBox)

    return [1] # TODO: return bbox center X, Y, bbox width, height (scale is 0 ~ 1)

  def saveImage(self, img, path):
    cv2.imwrite(path, img)

  def saveLabel(self):
    # desired output
    # 58 [bbox center X] [bbox center Y] [bbox width] [bbox height]

  def run(self):
    print("load images")
    img_color = cv2.imread(self.rawDataFileFullPath_list[0], cv2.IMREAD_COLOR)
    print("convert grayimages")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    print("calc BBox")
    label = self.calcBBox(img_gray)
    print("save images")
    self.saveImage(img_color, path)
    print("save labels")
    self.saveLabel()

if __name__ == "__main__":
  autoLabeler = AutoLabeler()
  autoLabeler.run()
