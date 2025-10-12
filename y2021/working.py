# General Idea of what the code does:
# Reads in imagee with cv2
# Scans for boxes with the boxdetect package
# Then using the boxes, determines if they are checked
# Then finds the right box using ocr and cordanite math
# Will then add to a spreadsheet or whatever the company needs

# Imports
from boxdetect import config
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from boxdetect.pipelines import get_boxes
from boxdetect.pipelines import get_checkboxes
import cv2
import pytesseract
import numpy as np
import time
import os



# Getting image

file_name = r'C:\Users\choll\PycharmProjects\MommyAI\testset\StPete\image (49).png'
imagecv = cv2.imread(file_name)
# Bad variable managment!
originalimg = imagecv
superoriginalimage = cv2.imread(file_name)

# Greyscales image
imagecv = cv2.cvtColor(imagecv, cv2.COLOR_RGB2GRAY)

# Crops image to reduce boxes
y = 230
x = 1
h = 130
w = 425
# Creates a black background of the same shape as the inital image
mask = np.zeros(imagecv.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 220), (400, 350), 255, -1)

# Does some smart stuff to invert mask and other math
# Tutorial here https://stackoverflow.com/questions/29810128/opencv-python-set-background-colour/38516242

# load background (could be an image too)
bk = np.full(imagecv.shape, 255, dtype=np.uint8)  # white bk
# cv2.imshow("bgmask", bk)
# get masked foreground
fg_masked = cv2.bitwise_and(imagecv, imagecv, mask=mask)

# get masked background, mask must be inverted
mask = cv2.bitwise_not(mask)
bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

# combine masked foreground and masked background
final = cv2.bitwise_or(fg_masked, bk_masked)

imagecv = cv2.bitwise_or(fg_masked, bk_masked)
# Debug
# cv2.imshow("final", imagecv)
# cv2.waitKey(0)

# Creates a mask for what will still be shown
# More info in tutorial here https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
cv2.rectangle(mask, (0, 230), (400, 350), 255, -1)

cv2.imwrite("tempimg.png", imagecv)
# Save masked image to disk to make it better with get_boxes
cwd = os.getcwd()
tempimgpath = cwd + r"\tempimg.png"

cv2.imshow("masked", imagecv)
# imagecv = imagecv[y:y+h, x:x+w]
cv2.imshow('Image', originalimg)
cv2.waitKey(0)

# 18, 19
# Config for boxdetect, found from https://pypi.org/project/boxdetect/#usage-examples
cfg = config.PipelinesConfig()
cfg.width_range = (15,25)
cfg.height_range = (15,26)
cfg.scaling_factors = [10]
cfg.wh_ratio_range = (0.6, 1.4)
cfg.group_size_range = (0, 1)
cfg.dilation_iterations = 0
cfg.thickness = 2
#cfg.autoconfigure([(19, 19), (20, 20), (18, 18)])
# ,(18,19),(19,20),(19,18)
# Use rects not grouping rects
rects, grouping_rects, image, output_image = get_boxes(
    tempimgpath, cfg=cfg, plot=False)
# Debug
print(rects)
num = 1
# OCRs the inital image for finding cords of the boxes we want
results = pytesseract.image_to_data(imagecv, lang="eng", config="--psm 11")
print(results)
cordlist = []
# Loops through all the boxes found by get_boxes boxes stored in (x, y, w, h)
distancelist = []
checkedlist = []
for i in rects:
    # Debug
    print("Itteration", num)
    print(i)
    print(i[0])
    print(i[1])
    print(i[2])
    print(i[3])
    # y, x for cropping, boxfind
    # Tutorial used https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    y = i[1]
    x = i[0]
    h = i[3]
    w = i[2]

    distance = abs(i[0] - 288) + abs(i[1] - 288)

    distancelist.append(distance)
    # Crops inital image to the first box, will iterate through for all of them
    crop = imagecv[y:y + h, x:x + w]
    y = 0
    x = 0
    # Work in progress
    print(type(crop))
    print(crop)

    cv2.imwrite("crop.png", crop)
    # Removes the top, bottom, far left and far right rows and coloums,
    # Reason is becuase the black pixels from the box were showing up in the image and this is an easy way to remove then
    # Tutorial used https://note.nkmk.me/en/python-numpy-delete/
    # Top row delete
    new_crop = np.delete(crop, 0, 0)
    new_crop = np.delete(new_crop, -1, 0)
    new_crop = np.delete(new_crop, -1, 0)
    new_crop = np.delete(new_crop, 0, 1)
    new_crop = np.delete(new_crop, -1, 1)
    new_crop = np.delete(new_crop, -1, 1)
    # Thresholds image, any pixel above 245 in value gets turned to white else gets turned to black
    # Tutorial used https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
    ret, new_crop = cv2.threshold(new_crop, 240, 255, cv2.THRESH_BINARY)
    black = np.count_nonzero(new_crop == 0)
    print(new_crop)
    print("------------------------------")
    print("This point is ", distance, "away from the ideal point")
    print("------------------------------")
    print("------------------------------")
    print("There are ", black, "black pixels in the image")
    print("------------------------------")
    if black >= 15:
        print("Image number", num, "at", i, "is checked!")
        checkedlist.append(1)
    else:
        print("Image number", num, "at", i, "is not checked")
        checkedlist.append(0)
    # Debug
    cv2.imwrite("newcrop.png", new_crop)

    # Shows image and adds 1 to the debug counter
    cv2.imshow('Image', crop)
    cv2.waitKey(0)
    num = num + 1
print(rects)
# Stops program becuase the findcheckboxes dosnt work to well
cv2.imshow("output", output_image)
cv2.waitKey(0)
# List of how far away the top of each box is away from the point 288, 288
print(distancelist)
# Finds the lowest amount in that list
minval = min(distancelist)
# Then finds the position of that value in the list
minpos = distancelist.index(min(distancelist))
topleftpos = distancelist.index(min(distancelist))
# Debug
print("Min value is", minval, "at position", minpos)
# Now we know the top left box, we are looking for the 4 box pair and with this we should be able to find the rest
topleft = rects[minpos]
# Debug
print("topleft is ", topleft)

# Copy the rects array and turn to list so we can remove the value we already know
rectscopy = rects
# Use tolist numpy function
rectscopy = rectscopy.tolist()
# Debug
print(rectscopy)

# Debug
print(rectscopy)

'''
Finds top right
'''
# Create new list to add the y dist values to
ydistlist = []
# Go through older list and find diffrence in y values, adding them all to ydistlist
# Will return a list like [11, -14, 7, 24, 0, 12] or smth, lowest value is used at bottom left
# Also important to take the absoulte value so That is done on line 188
for i in rectscopy:
    # VERY IMPORTATNT THAT YOU SKIP THE POINT THAT IS THE TOPLEFT DO THIS
    if i == rectscopy[minpos]:
        print("Skipping this, debug 1", i)
        ydistlist.append(999)
        pass
    else:
        ydist = topleft[1] - i[1]
        print("ydist is ", ydist)
        ydistlist.append(ydist)
ydistlist = [abs(ele) for ele in ydistlist]
# Debug
print(ydistlist)
# Takes the lowest value of the diffrence in y values between topleft and the rest of the values
yminval = min(ydistlist)
# Finds it in the list
yminpos = ydistlist.index(min(ydistlist))
toprightpos = ydistlist.index(min(ydistlist))
# Debugs
print("Min value for topright is ", yminval, "at position", yminpos)
# Creates the bottom left [x, y, w, h] might be h, w but idk
topright = rects[yminpos]
# More print debugs
print("bottom left", topright)
print(rects)
print(rectscopy)

'''
Finds bottom left box
'''
xdistlist = []
for i in rectscopy:
    if i == rectscopy[minpos] or i == rectscopy[yminpos]:
        print("Skipping this, debug 2", i)
        xdistlist.append(999)
        pass
    else:
        xdist = topleft[0] - i[0]
        print("xdist is ", xdist)
        xdistlist.append(xdist)
xdistlist = [abs(ele) for ele in xdistlist]

print(xdistlist)
xminval = min(xdistlist)
# Finds it in the list
xminpos = xdistlist.index(min(xdistlist))
bottomleftpos = xdistlist.index(min(xdistlist))
print("Min value for bottomleft is ", xminval, "at position", xminpos)
bottomleft = rectscopy[xminpos]
print("bottomeleft is", bottomleft)

'''
Finds the bottom right box
'''
ydistlist = []

for i in rectscopy:
    # VERY IMPORTATNT THAT YOU SKIP THE POINT THAT IS THE TOPLEFT DO THIS
    if i == rectscopy[minpos] or i == rectscopy[yminpos] or i == rectscopy[xminpos]:
        print("Skipping this, debug 1", i)
        ydistlist.append(999)
        pass
    else:
        ydist = topright[1] - i[1]
        print("ydist is ", ydist)
        ydistlist.append(ydist)
ydistlist = [abs(ele) for ele in ydistlist]
print(ydistlist)
# Takes the lowest value of the diffrence in y values between topleft and the rest of the values
yminval = min(ydistlist)
# Finds it in the list
yminpos = ydistlist.index(min(ydistlist))
bottomrightpos = ydistlist.index(min(ydistlist))

# Debugs
print("Min value for bottomright is ", yminval, "at position", yminpos)
# Creates the bottom left [x, y, w, h] might be h, w but idk
bottomright = rects[yminpos]
# More print debugs
print("bottom right", bottomright)

from itertools import combinations


def checks(topleft, topright, bottomleft, bottomright, rectscopy):
    num = 1
    successlist = []
    # print(topleft, topright, bottomleft, bottomright, rectscopy)
    '''
    Correct for hx5
    topleft = [288, 292, 18, 18]
    topright = [337, 292, 19, 18]
    bottomleft = [288, 274, 18, 18]
    bottomright = [337, 275, 19, 18]


    '''
    for i in rectscopy:
        topleft = i
        templist = rectscopy.copy()
        templist.remove(topleft)
        print("a")
        for ii in templist:
            # finding bottom left
            if abs(i[0] - ii[0]) < 5:
                print("b")
                if 30 > i[1] - ii[1] > 0:
                    bottomleft = ii
                    templist.remove(bottomleft)
                    print("c")
                    # Going to bottom right
                    for iii in templist:
                        if abs(ii[1] - iii[1]) < 5:
                            if 50 > iii[0] - ii[0] > 0:
                                print("d")
                                bottomright = iii
                                # Going for top right
                                templist.remove(bottomright)
                                for iiii in templist:
                                    print("e")
                                    if abs(iiii[0] - iii[0]) < 5:
                                        print("ree")
                                        print("Topleft bottom left, bottom right", i, ii, iii, iiii)
                                        print(iiii[1] - iii[1])
                                        print(templist)
                                        if 50 > iiii[1] - iii[1] > 0:
                                            topright = iiii
                                            print("f")
                                            success = topleft, topright, bottomleft, bottomright
                                            successlist.append(success)

    print(successlist)
    print(num)
    num += 1
    # Checks to make sure that their is at least 4 boxes detected
    if len(rectscopy) < 4:
        print("Failed")
        failed = True
    # Checks Y value of topleft and right, makes sure they are pretty much the same
    if abs(topleft[1] - topright[1]) > 7:
        print("Failed 1")
        failed = True
    # Checks Y value of botttom and right, makes sure they are pretty much the same
    if abs(bottomleft[1] - bottomright[1]) > 7:
        print("Failed 2")
        failed = True
    # Checks X value of Topright and bottomright, makes sure they are about the same
    if abs(topright[0] - bottomright[0]) > 7:
        print("Failed 3")
        failed = True
    # Same thing for left side
    if abs(topleft[0] - bottomleft[0]) > 7:
        print("Failed 4")
        failed = True
    # Checks to make sure the top is actual top
    if topleft[1] - bottomleft[0] > 0:
        print("Failed 6")
        failed = True
    # Checks to make sure the top is actual top
    # totalchecked = int(checkedlist[topleftpos]) + int(checkedlist[bottomleftpos]) + int(
    #    checkedlist[bottomrightpos]) + int(checkedlist[toprightpos])
    # if totalchecked >= 3:
    #    print("failed 5")
    #    print(totalchecked)
    #    failed = True

    return successlist


print(type(rectscopy))
successlist = checks(topleft, topright, bottomleft, bottomright, rectscopy)

print("here")
print("List of successs", successlist)
print("Amount of successes", len(successlist))
try:
    success = successlist[0]
except:
    pass
if len(successlist) == 0:
    print("------------------------------")
    print("No successes found... exiting")
    exit()
    time.sleep(2)
    print("------------------------------")

if len(successlist) > 1:
    print("a")
    ycordlist = []
    for i in successlist:
        ycord = i[0][1]
        ycordlist.append(ycord)
    truesuccess = max(ycordlist)
    index = ycordlist.index(truesuccess)
    success = successlist[index]

print("-------Success!!!!!----------")
print(success)
print("AA")

'''
TL BL
TR BR  - hx4

TL BL
TR BR  - hx1



'''

'''
Correct output is
topleft = [297 291  18  19]
topright = [297 309  18  18]
bottomleft = [346 291  20  18]
bottomright = [346 309  20  18]
'''


def rectntext(topleft, topright, bottomleft, bottomright, originalimg):
    start = topleft[0], topleft[1]
    end = topleft[0] + topleft[2], topleft[1] + topleft[3]
    cv2.rectangle(originalimg, start, end, (0, 0, 255), 1)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (topleft[0], topleft[1])
    # fontScale
    fontScale = .5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    # Using cv2.putText() method
    originalimg = cv2.putText(originalimg, 'TL', org, font,
                              fontScale, color, thickness, cv2.LINE_AA)

    org = (topright[0], topright[1])
    originalimg = cv2.putText(originalimg, 'TR', org, font,
                              fontScale, color, thickness, cv2.LINE_AA)
    org = (bottomleft[0], bottomleft[1])
    originalimg = cv2.putText(originalimg, 'BL', org, font,
                              fontScale, color, thickness, cv2.LINE_AA)
    org = (bottomright[0], bottomright[1])
    originalimg = cv2.putText(originalimg, 'BR', org, font,
                              fontScale, color, thickness, cv2.LINE_AA)

    start = topright[0], topright[1]
    end = topright[0] + topright[2], topright[1] + topright[3]
    cv2.rectangle(originalimg, start, end, (0, 0, 255), 1)

    start = bottomleft[0], bottomleft[1]
    end = bottomleft[0] + bottomleft[2], bottomleft[1] + bottomleft[3]
    cv2.rectangle(originalimg, start, end, (0, 0, 255), 1)

    start = bottomright[0], bottomright[1]
    end = bottomright[0] + bottomright[2], bottomright[1] + bottomright[3]
    cv2.rectangle(originalimg, start, end, (0, 0, 255), 1)

    print(i)
    return originalimg


originalimg = cv2.imread(file_name)
topleft = success[0]
topright = success[1]
bottomleft = success[2]
bottomright = success[3]
topleft = success[2]
topright = success[3]
bottomleft = success[0]
bottomright = success[1]
print(success[0], success[1], success[2], success[3])
print("RIGHT HERE")
newimage = rectntext(topleft, topright, bottomleft, bottomright, originalimg)
cv2.imshow("Newimg", newimage)
cv2.waitKey(0)

print("here")

print("Topleft", topleft, "Topright", topright)
print("Bottomleft", bottomleft, "Bottomright", bottomright)
print(checkedlist)
print(rectscopy)

topleftpos = rectscopy.index(topleft)
toprightpos = rectscopy.index(topright)
bottomrightpos = rectscopy.index(bottomright)
bottomleftpos = rectscopy.index(bottomleft)

if checkedlist[topleftpos] == 1:
    print("The patient has NOT had breast cancer")
if checkedlist[toprightpos] == 1:
    print("The patient HAS had breast cancer")
if checkedlist[bottomleftpos] == 1:
    print("The patient has NOT had biopsy or surgery")
if checkedlist[bottomrightpos] == 1:
    print("The patient HAS had a biopsy or surgery")

print(rectscopy)

cv2.imshow("Done??", originalimg)
cv2.imshow("Original", superoriginalimage)

cv2.waitKey(0)
exit()

plt.figure(figsize=(20, 20))
plt.imshow(output_image)
plt.show()

checkboxes = get_checkboxes(
    output_image, cfg=cfg, px_threshold=0.01, plot=False, verbose=True)
print(checkboxes)
print("Output object type: ", type(checkboxes))
for checkbox in checkboxes:
    print("Checkbox bounding rectangle (x,y,width,height): ", checkbox[0])
    print("Result of `contains_pixels` for the checkbox: ", checkbox[1])
    print("Display the cropout of checkbox:")
    plt.figure(figsize=(1, 1))
    plt.imshow(checkbox[2])
    plt.show()