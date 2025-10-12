import cv2
import numpy as np
import time
import os



directory = r"C:\Users\choll\PycharmProjects\MommyAI\testset\StPete"
os.chdir(directory)
for file in os.listdir(directory):
    filedir = os.path.join(directory, file)
    print(filedir)
    print(file)
    image = cv2.imread(filedir)
    height, width, _ = image.shape
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (0, 160), (550, 600), 255, -1)

    # Does some smart stuff to invert mask and other math
    # Tutorial here https://stackoverflow.com/questions/29810128/opencv-python-set-background-colour/38516242

    # load background (could be an image too)
    bk = np.full(image.shape, 255, dtype=np.uint8)  # white bk
    #cv2.imshow("bgmask", bk)
    # get masked foreground
    fg_masked = cv2.bitwise_and(image, image, mask=mask)

    # get masked background, mask must be inverted
    mask = cv2.bitwise_not(mask)
    bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

    # combine masked foreground and masked background

    crop = cv2.bitwise_or(fg_masked, bk_masked)
    print(crop.shape, image.shape)
    file = file.replace(" ", "")
    print("File name", file)

    cv2.imwrite(file, crop)
    #os.remove(filedir)


    # Crops inital image to the first box, will iterate through for all of them
