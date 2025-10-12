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
for image_path in os.listdir(folder):

    if image_path[-3:] == "png":
        continue
    if image_path[-3:] == "pdf":
        continue

    try:
        ds = dicom.dcmread(image_path)
    except:
        print(image_path, "Can not be scanned")
        totalfail += 1
        total += 1
        continue

    try:
        print(ds.DocumentTitle, image_path)
    except:
        print(image_path, "has no document title")
        totaltitle += 1
        #total += 1

    try:
        if ds.EncapsulatedDocument:
            pass

    except:
        print(image_path, "has no document at ALL")
        nodoccount += 1
    try:
        if ds.DocumentTitle[:2] == "HX" or "HI":
            HIHX += 1
    except:
        pass

    total += 1
    #print(ds)

    #cv2.imshow("Dicompdf", dicompdf)
    #cv2.waitKey(0)

print(totalfail, "Could not be scanned", totaltitle, "Did not have a title", total, "There was that many total")
print(nodoccount)
print(HIHX)