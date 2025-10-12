import cv2
import pytesseract
from PIL import Image
import os
import time
from pytesseract import Output
import pandas as pd

import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
image = ""
images = []
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\choll\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
imgpath = "C:\\Users\\choll\\PycharmProjects\\MommyAI\\Images\\HXSheets\\"
print(imgpath)

hx1 = cv2.imread(imgpath + "hxsheet1.png")
hx2 = cv2.imread(imgpath + "hxsheet2.png")
hx3 = cv2.imread(imgpath + "hxsheet3.png")
hx4 = cv2.imread(imgpath + "hxsheet4.png")
hx5 = cv2.imread(imgpath + "hxsheet5.png")
hx6 = cv2.imread(imgpath + "hxsheet6.png")
hx7 = cv2.imread(imgpath + "hxsheet7.png")
hx8 = cv2.imread(imgpath + "hxsheet8.png")
hx9 = cv2.imread(imgpath + "hxsheet9.png")
hx10 = cv2.imread(imgpath + "hxsheet10.png")


def imageconverting(image):

    #print(string)






    image = image[330:570, 0:350]
    #image = image[320:580, 0:500]
    oldimage = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(image)
    print(angle)
    image = rotate(image, angle, resize=True) * 255
    #cv2.imwrite("Rotimg.png", rotimage)
    cv2.imwrite("Normalimg.png", image)
    #image = cv2.imread("Normalimg.png")
    image = Image.fromarray(image)
    image = image.convert("L")
    image = np.array(image)
    arr = np.asarray(image)
    tot = arr.sum()
    print(tot)
    print(type(image))
    string = pytesseract.image_to_string(image, lang="eng", config="--psm 11")
    if "Breast Implants" in string:
        print("String found")

    if "LT" in string:
        print("LT String found")

    if "RT" in string:
        print("RT String found")

    results = pytesseract.image_to_data(image, lang="eng", config="--psm 11", output_type="data.frame")
    df = pd.DataFrame(results)
    #print(df)
    var = df.loc[df['text'] == 'Implants']
    #print(var)
    print(type(var))
    print()
    x,y,w,h = var["left"].values[0], var["top"].values[0], var["width"].values[0], var["height"].values[0]
    print(x, y, w, h)
    ychange = 0
    ynegchange = 0
    if y > 26:
        ychange = y - 26
    if y < 26:
        ynegchange = 26 - y
    boxX, boxY = x + 164, y - 5
    #hIMG, wIMG = image.shape

    #cv2.rectangle(image, (boxX, boxY), (293, 225 + ychange - ynegchange), (0,0,255),1)
    #cv2.rectangle(image, (x, y), (x + w, h + y), (0, 0, 255), 1)
    image = image[boxY:225 + ychange - ynegchange, boxX:300]
    #print(df.text.to_string(index=False))


    #rotatedimg = pytesseract.image_to_osd(image)
    #print(rotatedimg)

    print(boxX, boxY)
    image = cv2.bitwise_not(image)
    (thresh, image) = cv2.threshold(image, 11, 255, cv2.THRESH_BINARY)
    arr = np.asarray(image)
    tot = arr.sum()
    print("Final total", tot)
    cv2.imshow("HXsheet", image)
    cv2.waitKey(0)

imagegroup = [hx1, hx2, hx3, hx4, hx5, hx6, hx7, hx8, hx9, hx10]
for image in imagegroup:
    imageconverting(image)

#imageconverting(hx10)