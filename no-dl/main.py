import os
import numpy as np
import cv2 as cv
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

# Creating a plate dictionary
img_path = r'data\images'
img_dict = {}
i = 240
for img in os.listdir(img_path):
    img_dict[i] = os.path.join(img_path,img)
    i += 1
    
file_object = open('plates.txt', 'a')

# Parsing through plate dictionary
for key in img_dict:
    img = cv.imread(img_dict[key])
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # edge detection
    canny = cv.Canny(grey, 150, 175)

    # find contours based on edges
    contours, hierarchies  = cv.findContours(canny.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:30]

    # find the contour with 4 potential corners and create ROI around it
    for contour in contours:
            # find Perimeter of contour and it should be a closed contour
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.01 * perimeter, True)
            if len(approx) == 4:
                contour_with_license_plate = approx
                x, y, w, h = cv.boundingRect(contour)
                license_plate = grey[y:y + h, x:x + w]
                break
    (thresh, license_plate) = cv.threshold(license_plate, 127, 255, cv.THRESH_BINARY)

    # text recognition
    text = pytesseract.image_to_string(license_plate, config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # draw license plate and write the Text
    image = cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3) 
    image = cv.putText(img, text, (x-100,y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)

    cv.imshow("image", image)
    cv.waitKey(0)
    
    file_object.write(text)

file_object.close()