import pydicom as dicom
import matplotlib.pylab as plt
import time
import cv2
import os
from pdf2image import convert_from_path
# specify your image path
folder = r'C:\Users\choll\PycharmProjects\MommyAI\FinalTest'
os.chdir(folder)
totaltitle = 0
totalfail = 0
total = 0
nodoccount = 0
HIHX = 0
print(os.listdir(r"C:\Users\choll\PycharmProjects\MommyAI\LongHX"))
for image_path in os.listdir(folder):
    tempimagepath = image_path + ".png"
    ds = dicom.dcmread(image_path, force=True)
    try:
        print(ds.DocumentTitle)
    except:
        print(image_path, "has no document title")
        totaltitle += 1
        total += 1
        continue

    if ds.DocumentTitle[:2] == "HX":
        if tempimagepath in os.listdir(r"C:\Users\choll\PycharmProjects\MommyAI\LongHX"):
            HIHX += 1
            print(image_path)
            print("HIHX UP 1")
    if ds.DocumentTitle[:2] == "HI":
        if tempimagepath in os.listdir(r"C:\Users\choll\PycharmProjects\MommyAI\LongHX"):
            HIHX += 1
            print(image_path)
            print("HIHX UP 1")


    total += 1
    #print(ds)

    #cv2.imshow("Dicompdf", dicompdf)
    #cv2.waitKey(0)

print(totalfail, "Could not be scanned", totaltitle, "Did not have a title", total, "There was that many total")
print(nodoccount)
print(HIHX)