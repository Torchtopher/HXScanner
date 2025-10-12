# General Idea of what the code does:
# Reads in image with cv2
# Scans for boxes with the boxdetect package
# Then using the boxes, determines if they are checked
# Then finds the right box using ocr and coordinate math
# Will then add to a spreadsheet or whatever the company needs

# Imports
from boxdetect import config
import matplotlib.pyplot as plt
from boxdetect.pipelines import get_boxes
from boxdetect.pipelines import get_checkboxes
import cv2
import argparse
import numpy as np
import time
import os
from pathlib import Path

cwd = os.getcwd()
#parser = argparse.ArgumentParser(description='Reads HX sheets, add folder of images to be processed.')
#parser.add_argument('folder', type=Path, help='The folder to scan, relative or full path accepted')
#parser.add_argument('--od', '--outputDirectory',
#                    help='The place to output the list of files, outputs to current directory if left blank')
#parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
#                    help="increase output verbosity")
#args = parser.parse_args()
#verbosity = args.verbosity
#print(verbosity)
#folder = args.folder
#print(folder)

#outputdir = args.od
#print(outputdir)
print(cwd)
#folder = os.path.join(cwd, folder)
#print(folder)

# Getting image


file_name = r'C:\Users\choll\PycharmProjects\MommyAI\FinalTest\Augusta5.dcm'

imagecv = cv2.imread(file_name)
# Bad variable management!
originalimg = imagecv
superoriginalimage = cv2.imread(file_name)

# Greyscales image
imagecv = cv2.cvtColor(imagecv, cv2.COLOR_RGB2GRAY)
width = 630
height = 820
dim = (width, height)

# resize image
#imagecv = cv2.resize(imagecv, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite("out1.png", imagecv)
cv2.imwrite("tempimg.png", imagecv)
# Denoise image
# imagecv = cv2.fastNlMeansDenoising(imagecv, None, 30, 7, 21)
# Crops image to reduce boxes




##y = 230
##x = 1
##h = 130
##w = 425
### Creates a black background of the same shape as the inital image
##mask = np.zeros(imagecv.shape[:2], dtype="uint8")
##cv2.rectangle(mask, (0, 220), (400, 350), 255, -1)
##
### Does some smart stuff to invert mask and other math
### Tutorial here https://stackoverflow.com/questions/29810128/opencv-python-set-background-colour/38516242
##
### load background (could be an image too)
##bk = np.full(imagecv.shape, 255, dtype=np.uint8)  # white bk
### cv2.imshow("bgmask", bk)
### get masked foreground
##fg_masked = cv2.bitwise_and(imagecv, imagecv, mask=mask)
##
### get masked background, mask must be inverted
##mask = cv2.bitwise_not(mask)
##bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
##
### combine masked foreground and masked background
##final = cv2.bitwise_or(fg_masked, bk_masked)
##
##imagecv = cv2.bitwise_or(fg_masked, bk_masked)
### Debug
### cv2.imshow("final", imagecv)
### cv2.waitKey(0)
##
### Creates a mask for what will still be shown
### More info in tutorial here https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
##cv2.rectangle(mask, (0, 230), (400, 350), 255, -1)
##
##cv2.imwrite("tempimg.png", imagecv)
### Save masked image to disk to make it better with get_boxes
##
tempimgpath = cwd + r"\tempimg.png"
##
##cv2.imshow("masked", imagecv)
### imagecv = imagecv[y:y+h, x:x+w]
##cv2.imshow('Image', originalimg)
##cv2.waitKey(0)




# 18, 19
# Config for boxdetect, found from https://pypi.org/project/boxdetect/#usage-examples

cfg1 = config.PipelinesConfig()

cfg1.width_range = (26, 40)
cfg1.height_range = (26, 40)
cfg1.scaling_factors = [1]
cfg1.wh_ratio_range = (0.6, 1.4)
cfg1.group_size_range = (0, 1)
cfg1.dilation_iterations = 0
cfg1.thickness = 2

rects1, grouping_rects, image, output_image = get_boxes(
    tempimgpath, cfg=cfg1, plot=False)
try:
    rects1 = rects1.tolist()

except:
    print("Second pass found no boxes")
    rects1 = []

print(rects1)
cv2.imshow("Inital", output_image)
cv2.waitKey()
rectscopy = rects1
print(type(rectscopy))
# Use tolist numpy function
#rectscopy = rectscopy.tolist()
# want to know how many times I appened a cord
appendedcords = 0
print("Look here!", rects1, "and" ,rectscopy)

# Debug

num = 1

cordlist = []
# Loops through all the boxes found by get_boxes boxes stored in (x, y, w, h)
distancelist = []
checkedlist = []
for i in rectscopy:
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
    new_crop = np.delete(new_crop, 0, 0)
    new_crop = np.delete(new_crop, -1, 0)
    new_crop = np.delete(new_crop, -1, 0)
    new_crop = np.delete(new_crop, 0, 1)
    new_crop = np.delete(new_crop, -1, 1)
    new_crop = np.delete(new_crop, -1, 1)
    new_crop = np.delete(new_crop, 0, 0)
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
    if black >= 30:
        print("Image number", num, "at", i, "is checked!")
        checkedlist.append(1)
    else:
        print("Image number", num, "at", i, "is not checked")
        checkedlist.append(0)
    # Debug
    cv2.imwrite("newcrop.png", new_crop)

    # Shows image and adds 1 to the debug counter
    # cv2.imshow('Image', crop)
    # cv2.waitKey(0)
    num = num + 1

# Stops program becuase the findcheckboxes dosnt work to well
cv2.imshow("output", output_image)
cv2.imwrite("boxoutput.png", output_image)
cv2.waitKey(0)

from itertools import combinations


def checks(rectscopy):
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
        for ii in templist:
            # finding bottom left
            if abs(i[0] - ii[0]) < 5:
                if 60 > i[1] - ii[1] > 0:
                    bottomleft = ii
                    templist.remove(bottomleft)
                    # Going to bottom right
                    print("a")
                    for iii in templist:
                        if abs(ii[1] - iii[1]) < 5:
                            if 100 > iii[0] - ii[0] > 0:
                                print("aaa")
                                bottomright = iii
                                # Going for top right
                                templist.remove(bottomright)
                                print("b")
                                for iiii in templist:
                                    if abs(iiii[0] - iii[0]) < 5:
                                        print("c")
                                        if 60 > iiii[1] - iii[1] > 0:
                                            topright = iiii
                                            print("f")
                                            success = topleft, topright, bottomleft, bottomright
                                            successlist.append(success)

    print(successlist)
    print(num)
    num += 1

    return successlist


print(type(rectscopy))
successlist = checks(rectscopy)

print("here")
print("List of success", successlist)
print("Amount of successes", len(successlist))
print("Rectscopy", rectscopy)
try:
    success = successlist[0]
except:
    pass
logs = open("log.txt", "a+")
message = "\n" + str(rectscopy)
logs.write(str(message))
logs.close()
if len(successlist) == 0:
    print("------------------------------")
    print("No successes found... exiting")

    time.sleep(2)
    print("------------------------------")

if len(successlist) > 1:
    print("a")
    ycordlist = []
    for i in successlist:
        ycord = i[0][1]
        ycordlist.append(ycord)
    num = 0
    for i in ycordlist:
        if i > 750:
            ycordlist[num] = 0
        num += 1

    truesuccess = max(ycordlist)
    print(truesuccess)

    #442
    if truesuccess < 500:
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        exit()
        time.sleep(11)
    print("---------------------weeeeooooweeeoooooooo")
    print(truesuccess)
    print("---------------------weeeeooooweeeoooooooo")
    #245
    index = ycordlist.index(truesuccess)
    success = successlist[index]

print("-------Success!!!!!----------")
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
#newimage = rectntext(topleft, topright, bottomleft, bottomright, originalimg)
newimage = originalimg
for i in successlist:
    topleft = i[2]
    topright = i[3]
    bottomleft = i[0]
    bottomright = i[1]
    newimage = rectntext(topleft, topright, bottomleft, bottomright, newimage)

cv2.imshow("Newimg", newimage)
cv2.imwrite("ImageLabled.png", newimage)
cv2.waitKey(0)

print("here")
#[610, 382 upper box for dicom2
# 610 
print("Topleft", topleft, "Topright", topright)
print("Bottomleft", bottomleft, "Bottomright", bottomright)
print(checkedlist)
print(rectscopy)
x = 0
checkedlist = []
for i in success:
    y = i[1]
    x = i[0]
    h = i[3]
    w = i[2]


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
    new_crop = np.delete(new_crop, 0, 0)
    new_crop = np.delete(new_crop, -1, 0)
    new_crop = np.delete(new_crop, -1, 0)
    new_crop = np.delete(new_crop, 0, 1)
    new_crop = np.delete(new_crop, -1, 1)
    new_crop = np.delete(new_crop, -1, 1)
    new_crop = np.delete(new_crop, 0, 0)
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
    print("There are ", black, "black pixels in the image")
    print("------------------------------")

    if black >= 30:
        print("Image number", num, "at", i, "is checked!")
        checkedlist.append(1)
    else:
        print("Image number", num, "at", i, "is not checked")
        checkedlist.append(0)
    # Debug
    cv2.imwrite("newcrop.png", new_crop)
    cv2.imshow("newcrop.png", new_crop)
    cv2.waitKey(0)
    topleftcheck = False
    toprightcheck = False

    bottomleftcheck = False
    bottomrightcheck = False
    if x == 0:
        topleftcheck = True
    if x == 1:
        toprightcheck = True
    if x == 2:
        bottomleftcheck = True
    if x == 3:
        bottomrightcheck = True
    x += 1


topleftpos = 0
toprightpos = 1
bottomrightpos = 2
bottomleftpos = 3
cancer = 0
surgery = 0
if topleftcheck == True:
    print("The patient has NOT had breast cancer")
    cancer += 1
if toprightcheck == True:
    print("The patient HAS had breast cancer")
    cancer += 1
if bottomleftcheck == True:
    print("The patient has NOT had biopsy or surgery")
    surgery += 1
if bottomrightcheck == True:
    print("The patient HAS had a biopsy or surgery")
    surgery += 1

if cancer > 1:
    print("Failed, 2 boxes appear to be checked")
if surgery > 1:
    print("Failed, 2 boxes appear to be checked")
print("Appended", appendedcords, "coordinate")

cv2.imshow("Done??", originalimg)
superoriginalimage = rectntext(success[2], success[3], success[0], success[1], superoriginalimage)
cv2.imshow("Original", superoriginalimage)

cv2.waitKey(0)
exit()

