from boxdetect import config
from boxdetect.pipelines import get_boxes
import argparse
from pathlib import Path
import os
import pydicom as dicom
from pdf2image import convert_from_path
import cv2
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser(description='Reads HX sheets, add folder of images to be processed.')
    parser.add_argument('folder', type=Path, help='The folder to scan, relative or full path accepted')
    parser.add_argument('--o', '--output',
                        help='The name of the output file, defaults to HXout')
    args = parser.parse_args()

    folder = os.path.join(os.getcwd(), args.folder)
    output_name = "HXout.csv"
    if args.o:
        output_name = args.o + ".csv"
    print(f"Folder={folder} \nOutput name={output_name}")

    return folder, output_name

def readDicom(file, folder):
    dcmPath = os.path.join(folder, file)
    #print(f"Path to DCM is {dcmPath}")
    try:
        ds = dicom.dcmread(dcmPath)
        with open('temp.pdf', 'wb') as fp:
            fp.write(ds.EncapsulatedDocument)
    except:
        return False

    patientID = ds.PatientID
    DOB = ds.PatientBirthDate
    StudyUID = ds.StudyInstanceUID
    StudyDate = ds.StudyDate

    return patientID, DOB, StudyUID, StudyDate

def pdf2png():
    poppler_path = os.path.join(os.getcwd(), "poppler-0.68.0_x86/poppler-0.68.0/bin")
    pages = convert_from_path('temp.pdf', 150, poppler_path=poppler_path)

    for page in pages:
        page.save('tempdicom.png', 'PNG')


def boxDetect(file):
    # Detects boxes
    cfg1 = config.PipelinesConfig()
    cfg1.width_range = (30, 40)
    cfg1.height_range = (30, 40)
    cfg1.scaling_factors = [3]
    cfg1.wh_ratio_range = (0.6, 1.4)
    cfg1.group_size_range = (0, 1)
    cfg1.dilation_iterations = 0
    cfg1.thickness = 2
    tempimgpath = "tempdicom.png"
    rects, grouping_rects, image, output_image = get_boxes(
        tempimgpath, cfg=cfg1, plot=False)

    #cv2.imwrite(f"{file}.png", output_image)
    # If it can not find boxes returns False


    if len(rects) < 4:
        return True

    rects = rects.tolist()
    # Removes all detections that have a Y value under 380 or over 800 and X values over 950 or less than 450
    rects = [x for x in rects if x[1] < 800 and 950 > x[0] > 450]

    # Impossible to have the 5 5 split
    if len(rects) < 10:
        return False
    leftLine = []
    rightLine = []
    middleLine = 0
    # Finds mid line
    for i in rects:
        middleLine += i[0]
    middleLine = int(middleLine / len(rects))

    for i in rects:
        if i[0] <= middleLine:
            leftLine.append(i)
        else:
            rightLine.append(i)

    # Removes top detections then reevaluates mid line
    while len(leftLine) > 6:
        leftLine = removeHighest(leftLine)

    while len(rightLine) > 6:
        rightLine = removeHighest(rightLine)

    middleLine = 0
    for i in rightLine:
        middleLine += i[0]
    for i in leftLine:
        middleLine += i[0]
    middleLine = int(middleLine / (len(rightLine) + len(leftLine)))

    listProblem = True

    while listProblem == True:
        leftLen = len(leftLine)
        rightLen = len(rightLine)
        # Checks list length is equal and also equal to 5 or 6
        if leftLen == rightLen and leftLen == 5 or leftLen == rightLen and leftLen == 6:
            # Makes sure each box on left has box on right at close Y value
            if listCheck(leftLine, rightLine):
                listProblem = False
            else:
                # If it is 6 tries removing top two and retries
                if len(leftLine) == 6:
                    leftLine = removeHighest(leftLine)
                    rightLine = removeHighest(rightLine)
                    continue
                break
        else:
            # If left is longer than right, removes top left one
            if len(leftLine) > len(rightLine) and len(leftLine) - 1 >= 5:
                leftLine = removeHighest(leftLine)
                continue
            # Same thing for right
            elif len(rightLine) > len(leftLine) and len(rightLine) - 1 >= 5:
                rightLine = removeHighest(leftLine)
                continue
            else:
                break

    #visualize(leftLine, rightLine, middleLine, output_image)
    if listProblem == True:
        return False

    return leftLine, rightLine

# Takes in two sets of [x, y, w, h] a middle line (could just be left out) and a output image
def visualize(leftLine, rightLine, middleLine, output_image):
    x1, y1 = middleLine, 0
    x2, y2 = middleLine, 2000
    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    thickness = 1
    color = (0, 0, 255)
    for i in leftLine:
        org = (i[0], i[1])
        text = f'{i[0]} {i[1]}'
        output_image = cv2.putText(output_image, text, org, font, fontScale,
                                   color, thickness, cv2.LINE_AA, False)
    color = (0, 0, 0)
    for i in rightLine:
        org = (i[0], i[1])
        text = f'{i[0]} {i[1]}'
        output_image = cv2.putText(output_image, text, org, font, fontScale,
                                   color, thickness, cv2.LINE_AA, False)
    #cv2.imshow("Output", output_image)
    #cv2.waitKey(0)

def removeHighest(rects):
    rects.sort(key=lambda x: int(x[1]))
    rects.pop(0)
    return rects
# Input list of bounding boxes in the form [x, y, w, h] will return the set of four boxes
# That are correct for the history sheets

# Sorts list by second element then checks if that second element is close in both lists
def listCheck(leftLine, rightLine):
    leftLine.sort(key=lambda x: int(x[1]))
    rightLine.sort(key=lambda x: int(x[1]))
    check = True
    for idx, value in enumerate(leftLine):
        if abs(value[1] - rightLine[idx][1]) < 5:
            pass
        else:
            check = False
    return check

def isChecked(checkBoxes):
    checkList = []
    imagecv = cv2.imread("tempdicom.png")
    for i in checkBoxes:
        y = i[1]
        x = i[0]
        h = i[3]
        w = i[2]

        # Crops inital image to the first box,
        crop = imagecv[y:y + h, x:x + w]

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
        ret, new_crop = cv2.threshold(new_crop, 235, 255, cv2.THRESH_BINARY)
        black = np.count_nonzero(new_crop == 0)

        if black >= 150:
            checkList.append(1)
        else:
            checkList.append(0)

    return checkList

def output(file, output_name, dicomFailed=False, boxFailed=False, notHX=False, cancer=False, surgery=False, checkBoxProblem=False, args=None):
    if args is None:
        args = []
    output_path = os.path.join(os.getcwd(), output_name)
    if not os.path.exists(output_path):
        f = open(output_name, "a+")
        f.write("FileName,Result,Cancer,Surgery,DicomReadFail,BoxesFailed,NotHX,Patient_ID,DOB,Study_Instance_UID,StudyDate,MultiCheckBoxes\n")
        f.close()
        return
    if file == None:
        return
    if dicomFailed == True:
        f = open(output_name, "a+")
        f.write(f"{file},-1,-1,-1,{dicomFailed},-1,-1,-1,-1,-1,-1,-1\n")
        f.close()

    elif boxFailed == True:
        f = open(output_name, "a+")
        f.write(f"{file},-1,-1,-1,-1,{boxFailed},{notHX},{args[0]},{args[1]},{args[2]},{args[3]},-1\n")
        f.close()

    elif notHX == True:
        f = open(output_name, "a+")
        f.write(f"{file},-1,-1,-1,-1,-1,{notHX},{args[0]},{args[1]},{args[2]},{args[3]},-1\n")
        f.close()

    elif cancer or surgery and not checkBoxProblem:
        f = open(output_name, "a+")
        f.write(f"{file},0,{cancer},{surgery},-1,-1,{notHX},{args[0]},{args[1]},{args[2]},{args[3]},-1\n")
        f.close()
    elif checkBoxProblem:
        f = open(output_name, "a+")
        f.write(f"{file},-1,-1,-1,-1,-1,{notHX},{args[0]},{args[1]},{args[2]},{args[3]},{checkBoxProblem}\n")
        f.close()
    else:
        # Souce of 0s 
        f = open(output_name, "a+")
        f.write(f"{file},1,0,0,-1,-1,{notHX},{args[0]},{args[1]},{args[2]},{args[3]},-1\n")
        f.close()



def main():
    counter = 0
    folder, output_name = parseArgs()
    output(None, output_name)
    with open(f"{output_name}", "r") as results:
        resultsData = results.read()

    for file in os.listdir(folder):
        counter += 1
        if file in resultsData:
            print(f"File already scanned! {file}")
            continue

        if file[-3:] == "dcm":
            try:
                patientID, DOB, StudyUID, StudyDate = readDicom(file, folder)
            except:
                print(f"Dicom reading for {file} has failed")
                output(file, output_name, dicomFailed=True)
                continue
        else:
            print(f"{file} is not a .dcm file")
            continue
        # Converts pdf to png
        pdf2png()
        args = [patientID, DOB, StudyUID, StudyDate]
        try:
            leftSide, rightSide = boxDetect(file)
        except:
            # Thinks its not a history form
            if boxDetect(file) == True:
                output(file, output_name, notHX=True, args=args)
            # Can not make boxes in 5 or 6 line
            elif boxDetect(file) == False:
                output(file, output_name, boxFailed=True, args=args)

            continue

        leftChecked = isChecked(leftSide)
        rightChecked = isChecked(rightSide)

        cancer = ''
        surgery = ''
        checkBoxProblem = False
        # No cancer/surgery
        if leftChecked[-1] + rightChecked[-1] == 1:
            if leftChecked[-1] == 1:
                surgery = False
            elif rightChecked[-1] == 1:
                surgery = True
        if leftChecked[-2] + rightChecked[-2] == 1:
            if leftChecked[-2] == 1:
                cancer = False
            elif rightChecked[-2] == 1:
                cancer = True
        if cancer == '' or surgery == '':
            checkBoxProblem = True
        if checkBoxProblem:
            output(file, output_name, checkBoxProblem=True, args=args)
        # No cancer, surgery or problems with checkbox
        elif not checkBoxProblem:
            output(file, output_name, cancer=cancer, surgery=surgery, args=args)

    print("-----------History Scanner Finished-----------")
    print(f"-----------{counter} Files Scanned!-----------")
    print(f"-----------Output located at {os.path.abspath(output_name)}-----------")


if __name__ == "__main__":
    print("Starting History Scanner")
    main()