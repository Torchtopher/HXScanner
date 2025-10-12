from pystackreg import StackReg
import cv2
from skimage import io
import matplotlib.pyplot as plt
from image_registration import chi2_shift
from skimage import io

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
image = cv2.resize(ref, dim, interpolation=cv2.INTER_AREA)
offset_image = cv2.resize(mov, dim, interpolation=cv2.INTER_AREA)

from skimage.feature import register_translation
shifted, error, diffphase = register_translation(image, offset_image, 100)
xoff = -shifted[1]
yoff = -shifted[0]


print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()