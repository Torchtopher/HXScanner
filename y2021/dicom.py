import pydicom as dicom
import matplotlib.pylab as plt
import time
import cv2
import os
from pdf2image import convert_from_path
# specify your image path
folder = r'C:\Users\choll\PycharmProjects\MommyAI\FinalTest'
os.chdir(folder)
for image_path in os.listdir(folder):
    if image_path[-3:] == "png":
        continue
    if image_path[-3:] == "pdf":
        continue
    finalname = image_path + ".png"
    if finalname in os.listdir(folder):
        print("Skiping", image_path)
        continue
    try:
        ds = dicom.dcmread(image_path)

    except:
        try:
            print("Failed orignal, tring force")
            ds = dicom.dcmread(image_path, force=True)
        except:
            print("Force failed")
            continue

    try:
        if ds.EncapsulatedDocument == True:
            pass
    except:
        print("No document detected")
        continue
    #print(ds)

    with open("dexample.pdf", 'wb') as fp:
        fp.write(ds.EncapsulatedDocument)

    poppler_path = r"C:\Users\choll\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin"
    pages = convert_from_path('dexample.pdf', 150, poppler_path=poppler_path)

    for page in pages:

        page.save(finalname, 'PNG')


    dicompdf = cv2.imread("Dicom3.png")
    print(image_path)
    #cv2.imshow("Dicompdf", dicompdf)
    #cv2.waitKey(0)