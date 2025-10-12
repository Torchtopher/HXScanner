from boxdetect import config
import matplotlib.pyplot as plt
from boxdetect.pipelines import get_boxes
import cv2
import argparse
import numpy as np
import os
from pathlib import Path
import pydicom as dicom
import logging
from pdf2image import convert_from_path
from collections import namedtuple
import datetime
# tesseract
import pytesseract
from pytesseract import Output

cfg1 = config.PipelinesConfig()

cfg1.width_range = (55,75)
cfg1.height_range = (55, 75)
cfg1.scaling_factors = [2]
cfg1.wh_ratio_range = (0.6, 1.4)
cfg1.group_size_range = (0, 1)
cfg1.dilation_iterations = 1
cfg1.thickness = 2


def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Directory of input dicoms")    
    ap.add_argument("-d", "--debug", required=False, help="Debug mode", default=False, action='store_true')
    args = vars(ap.parse_args())
    return args

'''
Takes in a path and returns a list of all the dcm files in the path
'''
def load_dcm_files(path):
    files = []
    dirpath = os.path.join(os.getcwd(), path)
    for file in os.listdir(path):
        if file.endswith(".dcm"):
            logging.debug(f"File {file}")
            try:
                ds = dicom.dcmread(os.path.join(dirpath, file))
                files.append({'dicom': ds, 'path': os.path.join(dirpath, file), 'image': None})
            except:
                logging.warning(f"Could not read file {os.path.join(dirpath, file)}")
                continue 
    return files

'''
takes some text and OCR dict and finds the coordinates of the first word of that text
I.e if text is "Have you had Breast Cancer?"
'''
def find_idx_of_text(unsplittext, ocr):
    splittext = unsplittext.split()

    for idx, text in enumerate(ocr['text']):
        for i in range(len(splittext)):
            if idx + i >= len(ocr['text']):
                break
            if ocr['text'][idx + i] != splittext[i]:
                break
        else:
            print(idx, text)
            x = ocr['left'][idx]
            y = ocr['top'][idx]
            w = ocr['width'][idx]
            h = ocr['height'][idx]
            break
    return (x, y, w, h)

'''
Does thresholding on the checkbox to determine if it is checked or not
returns number of black pixels
'''
def process_checkbox(checkbox):
    # run thresholding on the checkbox to turn it into a binary image
    ret_val, thresh = cv2.threshold(checkbox, 200, 255, cv2.THRESH_BINARY)
    # remove a 3 pixel border
    thresh = thresh[3:thresh.shape[0] - 3, 3:thresh.shape[1] - 3]

    black = np.count_nonzero(thresh == 0)
    return black

'''
appends a line to the csv file
'''
def write_csv(file_handle, dcmWpath, has_cancer, has_surgery, no_dicom=False):
    # write results to csv
    if not has_cancer and not has_surgery:
        useable = True
    else:
        useable = False
    if no_dicom:
        patientID = "N/A"
        DOB = "N/A"
        StudyUID = "N/A"
        StudyDate = "N/A"
        file_handle.write(f"{Path(dcmWpath['path']).stem},{has_cancer},{has_surgery},{useable},{patientID},{DOB},{StudyUID},{StudyDate}\n")
        return
    ds = dcmWpath['dicom']
    print(ds)
    patientID = ds.PatientID
    DOB = ds.PatientBirthDate
    StudyUID = ds.StudyInstanceUID
    StudyDate = ds.StudyDate
    file_handle.write(f"{Path(dcmWpath['path']).stem},{has_cancer},{has_surgery},{useable},{patientID},{DOB},{StudyUID},{StudyDate}\n")


def main():
    args = parse_args()
    
    if args['debug']:
        #logging.basicConfig(level=logging.DEBUG)
        log_dir = 'logs'
        # make log directory if it doesn't exist
        if not os.path.exists(os.path.join(os.getcwd(), log_dir)):
            os.makedirs(os.path.join(os.getcwd(), log_dir))
        # log based on current time 
        logging.basicConfig(filename=f'{log_dir}/HX_SCAN:{datetime.datetime.now()}.log', encoding='utf-8', level=logging.DEBUG)

    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug(f"Args {args}")

    '''
    Process goes as follows:
    1. Load DCM files
    2. Read the pdf encasulated in the DCM file
    3. Convert the pdf to an image (png)
    4. Run the box detection algorithm on the image to detect the boxes and which ones are checked
    5. Write the results to a csv file
    '''
    # DEBUG CASE
    if args['debug'] and os.path.exists(os.path.join(os.getcwd(), 'FastPath')):
        # if debug and FastPath dir exists, load the images from there
        dcm_files = []
        for file in os.listdir('FastPath'):
            if file.endswith(".png"):
                logging.debug(f"File {file}")
                try:
                    img = cv2.imread(os.path.join(os.getcwd(), 'FastPath', file), cv2.IMREAD_GRAYSCALE)
                    dcm_files.append({'dicom': None, 'path': os.path.join(os.getcwd(), 'FastPath', file), 'image': img})
                except:
                    logging.warning(f"Could not read file {os.path.join(os.getcwd(), 'FastPath', file)}")
                    continue
        logging.warn("Taking images from FastPath directory")
        print("Taking images from FastPath directory")
    # NORMAL CASE
    else:
        # 1. Load DCM files
        dcm_files = load_dcm_files(args['input'])
        #logging.debug(f"DCM Files {dcm_files}")

        # 2/3 Read the pdf encasulated in the DCM file and convert to image
        
        for dcmWpath in dcm_files:
            with open('to_convert_to_png.pdf', 'wb') as fp:
                try:
                    fp.write(dcmWpath['dicom'].EncapsulatedDocument)
                except:
                    logging.warning(f"{dcmWpath['path']} does not have an EncapsulatedDocument (pdf), skipping")
                    continue

            pdf = convert_from_path('to_convert_to_png.pdf', dpi=300, grayscale=True)
            os.remove('to_convert_to_png.pdf')
            if len(pdf) > 1:
                logging.warning(f"More than one page in {dcmWpath['path']}, skipping")
                continue

            for page in pdf:
                page.save('temp.png', 'PNG')
            img = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)
            os.remove('temp.png')
            print(img.shape)
            dcmWpath['image'] = img
    
    # remove everyone in dcm_files that doesn't have an image
    logging.debug(f"DCM Files {dcm_files}")
    # if debug and FastPath dir doesn't exist, make it
    if args['debug'] and not os.path.exists(os.path.join(os.getcwd(), 'FastPath')):
        os.makedirs(os.path.join(os.getcwd(), 'FastPath'))
        for dcmWpath in dcm_files:
            cv2.imwrite(f"FastPath/{Path(dcmWpath['path']).stem}.png", dcmWpath['image'])

    output_file = f"output{datetime.datetime.now()}.csv"
    # initalize csv
    with open(output_file, "w") as fp:
        fp.write("FileName,HasCancer,HasSurgery,Useable,Patient_ID,DOB,Study_UID,Study_Date\n")

    # run ocr on all the images
    total_success = 0
    for dcmWpath in dcm_files:
        img_to_str = pytesseract.image_to_string(dcmWpath['image'], lang='eng', config='--psm 6')
        dcmWpath['ocr'] = pytesseract.image_to_data(dcmWpath['image'], output_type=Output.DICT, lang='eng', config='--psm 6')     
        logging.debug(f"OCR {dcmWpath['ocr']}")
        print(dcmWpath['ocr'])
        BREAST_CANCER_STRING = "Have you had Breast Cancer?"
        BIOPSY_STRING = "Have you had biopsy or surgery?"

        if not (BREAST_CANCER_STRING in img_to_str) or not (BIOPSY_STRING in img_to_str):
            logging.warning(f"Could not find both questions in {dcmWpath['path']}, skipping")
            continue

        logging.debug(f"Found both questions in {dcmWpath['path']}")
        total_success += 1

        cancer_coords = find_idx_of_text(BREAST_CANCER_STRING, dcmWpath['ocr'])
        surgery_coords = find_idx_of_text(BIOPSY_STRING, dcmWpath['ocr'])

        if args['debug']:
            # draw boxes around the text
            start = cancer_coords[0], cancer_coords[1]
            end = cancer_coords[0] + cancer_coords[2], cancer_coords[1] + cancer_coords[3]
            cv2.rectangle(dcmWpath['image'], start, end, (0, 0, 255), 10)
            start = surgery_coords[0], surgery_coords[1]
            end = surgery_coords[0] + surgery_coords[2], surgery_coords[1] + surgery_coords[3]
            cv2.rectangle(dcmWpath['image'], start, end, (0, 0, 255), 10)
            
        # run box detection
        rects, _, _, _ = get_boxes(dcmWpath['image'], cfg1)

        # find two boxes that are in line with the text
        cancer_boxs = []
        surgery_boxs = []
        PIXEL_THRESHOLD = 30
        for box in rects:
            if abs(box[1] - cancer_coords[1]) < PIXEL_THRESHOLD:
                cancer_boxs.append(box)
            if abs(box[1] - surgery_coords[1]) < PIXEL_THRESHOLD:
                surgery_boxs.append(box)
        
        # if we didn't find the boxes, skip
        if len(cancer_boxs) != 2 or len(surgery_boxs) != 2:
            print("Could not find both boxes")
            logging.warning(f"Could not find both boxes in {dcmWpath['path']}, skipping")
            continue

        if args['debug']:
            # make cancer box and surgery box bigger with opencv
            for cancer_box in cancer_boxs:
                start = cancer_box[0], cancer_box[1]
                end = cancer_box[0] + cancer_box[2], cancer_box[1] + cancer_box[3]
                #cv2.rectangle(dcmWpath['image'], start, end, (0, 0, 255), 20)
            for surgery_box in surgery_boxs:
                start = surgery_box[0], surgery_box[1]
                end = surgery_box[0] + surgery_box[2], surgery_box[1] + surgery_box[3]
                #cv2.rectangle(dcmWpath['image'], start, end, (0, 0, 255), 20)

            for box in rects:
                print(box)
                start = box[0], box[1]
                end = box[0] + box[2], box[1] + box[3]
                #cv2.rectangle(dcmWpath['image'], start, end, (0, 0, 255), 10)
            
            # write text to file based on dcmWpath['path']
            with open(f"TestAlgo/{Path(dcmWpath['path']).stem}.txt", 'w') as fp:
                fp.write(str(dcmWpath['ocr'])) 
        
        # doesn't really matter which box is which, just get the one with smaller x value
        idx0_cancer_box = dcmWpath['image'][cancer_boxs[0][1]:cancer_boxs[0][1] + cancer_boxs[0][3], cancer_boxs[0][0]:cancer_boxs[0][0] + cancer_boxs[0][2]]
        idx1_cancer_box = dcmWpath['image'][cancer_boxs[1][1]:cancer_boxs[1][1] + cancer_boxs[1][3], cancer_boxs[1][0]:cancer_boxs[1][0] + cancer_boxs[1][2]]
        # get cancer box with smaller x value
        if cancer_boxs[0][0] < cancer_boxs[1][0]:
            no_cancer_box = idx0_cancer_box
            yes_cancer_box = idx1_cancer_box
        else:
            no_cancer_box = idx1_cancer_box
            yes_cancer_box = idx0_cancer_box

        idx0_surgey_box = dcmWpath['image'][surgery_boxs[0][1]:surgery_boxs[0][1] + surgery_boxs[0][3], surgery_boxs[0][0]:surgery_boxs[0][0] + surgery_boxs[0][2]]
        idx1_surgey_box = dcmWpath['image'][surgery_boxs[1][1]:surgery_boxs[1][1] + surgery_boxs[1][3], surgery_boxs[1][0]:surgery_boxs[1][0] + surgery_boxs[1][2]]
        # get surgery box with smaller x value
        if surgery_boxs[0][0] < surgery_boxs[1][0]:
            no_surgery_box = idx0_surgey_box
            yes_surgery_box = idx1_surgey_box
        else:
            no_surgery_box = idx1_surgey_box
            yes_surgery_box = idx0_surgey_box
        
        # process the boxes
        no_cancer_black = process_checkbox(no_cancer_box)
        yes_cancer_black = process_checkbox(yes_cancer_box)
        no_surgery_black = process_checkbox(no_surgery_box)
        yes_surgery_black = process_checkbox(yes_surgery_box)
        logging.debug(f"no_cancer_black: {no_cancer_black}, yes_cancer_black: {yes_cancer_black}, no_surgery_black: {no_surgery_black}, yes_surgery_black: {yes_surgery_black}")
        # if the number of black pixels in the yes box is 10 times greater than the no box, then the answer is yes
        has_cacner = yes_cancer_black > no_cancer_black * 10 
        has_surgery = yes_surgery_black > no_surgery_black * 10
        logging.debug(f"has_cacner: {has_cacner}, has_surgery: {has_surgery}")
        if args['debug']:
            # write results on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2
            color = (0, 0, 255)
            thickness = 2
            dcmWpath['image'] = cv2.cvtColor(dcmWpath['image'], cv2.COLOR_GRAY2BGR)

            if has_cacner:
                cv2.putText(dcmWpath['image'], 'Yes has cancer', (cancer_boxs[1][0], cancer_boxs[1][1]), font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(dcmWpath['image'], 'No cancer', (cancer_boxs[0][0], cancer_boxs[0][1]), font, fontScale, color, thickness, cv2.LINE_AA)
            if has_surgery:
                cv2.putText(dcmWpath['image'], 'Yes surgery', (surgery_boxs[1][0], surgery_boxs[1][1]), font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(dcmWpath['image'], 'No surgey', (surgery_boxs[0][0], surgery_boxs[0][1]), font, fontScale, color, thickness, cv2.LINE_AA)
            # show image 
            cv2.imshow('debug_image', dcmWpath['image'])
            cv2.waitKey(0)
            cv2.imwrite(f"TestAlgo/{Path(dcmWpath['path']).stem}.png", dcmWpath['image'])
        
        with open(output_file, "a") as fp:
            print(dcmWpath)
            write_csv(fp, dcmWpath, has_cacner, has_surgery, no_dicom=(dcmWpath['dicom'] is None))

if __name__ == '__main__':
    main()