import os

import scipy as sp
import scipy.misc
import imageio
import imreg_dft as ird
import cv2

import matplotlib.pyplot as plt
# the TEMPLATE

ref = cv2.imread(r'Images/HXSheets/hxsheet6.png')
mov = cv2.imread(r'Images/HXSheets/hxsheet3.png')
cv2.imwrite("ref.tiff", ref)
cv2.imwrite("mov.tiff", mov)
ref = cv2.imread(r'ref.tiff')
mov = cv2.imread(r'mov.tiff')


ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
mov = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)
width = 620
height = 815
dim = (width, height)
# resize image
im0 = cv2.resize(ref, dim, interpolation=cv2.INTER_AREA)
im1 = cv2.resize(mov, dim, interpolation=cv2.INTER_AREA)
result = ird.translation(im0, im1)
tvec = result["tvec"].round(4)
# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=tvec)
print(timg)
print(type(timg))
cv2.imwrite("Align4out.png", timg)
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":

    ird.imshow(im0, im1, timg)
    plt.show()

print("Translation is {}, success rate {:.4g}"
      .format(tuple(tvec), result["success"]))