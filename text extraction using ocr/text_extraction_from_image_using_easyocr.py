# -*- coding: utf-8 -*-
"""text extraction from image using easyocr.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/121fomnagJ1SatWxZ9lr1abPzP-2HhW4Y
"""

#installing and importing the necessary libraries
!pip install easyocr
import easyocr
import cv2

#mounting google drive to access the images
from google.colab import drive
drive.mount('/content/drive')
import os

#creating a function display to show the images in the output console
import matplotlib.pyplot as plt
def display(im_path):
    dpi=80
    im_data=plt.imread(im_path)
    height,width,depth=im_data.shape

    figsize=width/float(dpi), height/float(dpi)

    fig=plt.figure(figsize=figsize)
    ax=fig.add_axes([0,0,1,1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')

    plt.show()

#creating a function to read text from image using easyocr library functions
def recognize_text(img_path):
  reader = easyocr.Reader(['en'])
  return reader.readtext(img_path)

if __name__ == "__main__":
  from glob import glob
image = glob("/content/drive/My Drive/ocr images/*.[jJpP][pPnN][eE][gG]")

for im in image:
  display(im)
  result = recognize_text(im)
  for i in result:
    print(i[1])

# deskewing an image according to tilt of particular image
#import numpy as np
#def getSkewAngle(cvImage):
    # Prep image, copy, convert to gray scale, blur, and threshold
    #newImage = cvImage.copy()
    #gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (9, 9), 0)
    #thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    #dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    #contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key = cv2.contourArea, reverse = True)
    #for c in contours:
     #   rect = cv2.boundingRect(c)
      #  x,y,w,h = rect
       # cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    #largestContour = contours[0]
    #print (len(contours))
    #minAreaRect = cv2.minAreaRect(largestContour)
    #cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    #angle = minAreaRect[-1]
    #if angle < -45:
     #   angle = 90 + angle
    #return -1.0 * angle
# Rotate the image around its center
#def rotateImage(cvImage, angle):
 #   newImage = cvImage.copy()
 #   (h, w) = newImage.shape[:2]
 #   center = (w // 2, h // 2)
 #   M = cv2.getRotationMatrix2D(center, angle, 1.0)
 #   newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
 #   return newImage

# Deskew image
#def deskew(cvImage):
    #angle = getSkewAngle(cvImage)
    #return rotateImage(cvImage, 0.5*angle)