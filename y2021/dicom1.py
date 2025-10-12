import pydicom as dicom
import matplotlib.pylab as plt
import time
import cv2
import os
# specify your image path
image_path = r'C:\Users\choll\PycharmProjects\MommyAI\FinalTest\SouthShore2.dcm'
ds = dicom.dcmread(image_path)
print(ds.DocumentTitle)
print(ds)
with open('dexample.pdf', 'wb') as fp:
    fp.write(ds.EncapsulatedDocument)
from pdf2image import convert_from_path
poppler_path = r"C:\Users\choll\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin"
pages = convert_from_path('dexample.pdf', 150, poppler_path=poppler_path)
for page in pages:
    page.save('Dicom3.png', 'PNG')
dicompdf = cv2.imread("Dicom3.png")
cv2.imshow("Dicompdf", dicompdf)
cv2.waitKey(0)