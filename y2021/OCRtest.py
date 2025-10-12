import pytesseract
from PIL import Image
import cv2
from boxdetect.pipelines import get_boxes
from boxdetect import config

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\choll\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
imagepath = "C:\\Users\\choll\\PycharmProjects\\MommyAI\\Images\\HXSheets\\hxsheet1.png"
image = cv2.imread(imagepath)
string = pytesseract.image_to_string(image, lang="eng", config="--psm 11")
print(string)