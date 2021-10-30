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
    self.currentDirPath = os.getcwd() # /home/kcy/Auto_labeling_for_YOLOv5
    self.rawDataDirPath = self.currentDirPath + "/raw_data"
    self.rawDataFilePath_list = os.listdir(self.rawDataDirPath)
    self.rawDataFileFullPath_list = [self.rawDataDirPath + '/' + file_name for file_name in self.rawDataFilePath_list]

    self.imagesDirPath = self.currentDirPath +"/kcyoon/images"
    self.BBoxImagesDirPath = self.currentDirPath +"/kcyoon/bbox_images"
    self.labelsDirPath = self.currentDirPath +"/kcyoon/labels"

    self.dataCount = 0

  def showImage(self, img):
    cv2.imshow('img', img)
    cv2.waitKey(0)

  def calcBBox(self, img, criteria, imgCount):
    imgHeight_px, imgWidth_px = img.shape

    imgObject = np.where(img < criteria) # TODO: need to tune for stable functionality

    BBoxXMin_px = np.min(imgObject[1])
    BBoxXMax_px = np.max(imgObject[1])
    BBoxYMin_px = np.min(imgObject[0])
    BBoxYMax_px = np.max(imgObject[0])

    imgObjectBBox = cv2.rectangle(img, (BBoxXMin_px, BBoxYMin_px), (BBoxXMax_px, BBoxYMax_px), 100, 5)
    # self.showImage(imgObjectBBox)
    print("save bbox images")
    self.saveBBoxImage(imgObjectBBox, imgCount)

    BBoxCenterX_px = (BBoxXMin_px + BBoxXMax_px) / 2
    BBoxCenterY_px = (BBoxYMin_px + BBoxYMax_px) / 2
    BBoxWidth_px = BBoxXMax_px - BBoxXMin_px
    BBoxHeight_px = BBoxYMax_px - BBoxYMin_px

    return [BBoxCenterX_px/imgWidth_px, BBoxCenterY_px/imgHeight_px, BBoxWidth_px/imgWidth_px, BBoxHeight_px/imgHeight_px]

  def saveBBoxImage(self, img, imgCount):
    BBoximageFileFullPath = self.BBoxImagesDirPath + "/" + str(imgCount).zfill(12) + ".jpg"
    cv2.imwrite(BBoximageFileFullPath, img)

  def saveImage(self, img, imgCount):
    imageFileFullPath = self.imagesDirPath + "/" + str(imgCount).zfill(12) + ".jpg"
    cv2.imwrite(imageFileFullPath, img)

  def saveLabel(self, classNumber, BBoxData, labelCount):
    # desired output
    # 58 [bbox center X] [bbox center Y] [bbox width] [bbox height]
    # RAII Style Coding - keep away from user's mistakes
    labelFileFullPath = self.labelsDirPath + "/" + str(labelCount).zfill(12) + ".txt"
    strBBoxData = ""
    for element in BBoxData:
      strBBoxData += " " + str(element)
    with open(labelFileFullPath, 'w') as labelFile:
      labelFile.write(str(classNumber) + strBBoxData)

  def run(self):
    for rawDataPath in self.rawDataFileFullPath_list:
      print("load images")
      img_color = cv2.imread(rawDataPath, cv2.IMREAD_COLOR)
      print("convert grayimages")
      img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
      print("calc BBox")
      BBoxData = self.calcBBox(img_gray, 200, self.dataCount)
      print("save images")
      self.saveImage(img_color, self.dataCount)
      print("save labels")
      self.saveLabel(58, BBoxData, self.dataCount)
      print("{} data done!\n".format(self.dataCount))
      self.dataCount += 1

if __name__ == "__main__":
  autoLabeler = AutoLabeler()
  autoLabeler.run()
