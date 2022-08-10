# import cv2
# from matplotlib import pyplot as plt


# #Opening an Image
# image_file = 'op7.png'
# img = cv2.imread(image_file)
# # cv2.imshow("original image", img)

# #displaying diff images with actual size in matplotlib subplot
# def display(im_path):
#     dpi=80
#     im_data = plt.imread(im_path)
#     height, width, depth = im_data.shape

#     figsize = width / float(dpi), height/float(dpi)

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0,0,1,1])
    
#     #hide spins, ticks, etc
#     ax.axis('off')

#     ax.imshow(im_data, cmap = 'gray')
#     plt.show()

# #call the function to show image
# display(image_file)

# #INVERTED IMAGE


# import openpyxl
# import easyocr

# ## Easyocr Extraction:

# # initialize easyocr reader
# reader = easyocr.Reader(['en'], gpu=False)
# # read text from the image
# result = reader.readtext('/home/samyak/table_data_extract/cropped_output_images/output4.jpg')
# print(result)

# ## grouping the detections to list. It is nothing but i grouped the values with bounding boxes by 20 pixel variation. change C for different image and check it out. 

# flag = 0
# Main_list = []
# sub_list = []
# C = 20 # change value of C according to your image
# # first loop for get text value one by one value from result
# for i in range(len(result)):
#     # second for loop for check the text is with in flag range. For example if C value is 10, then check text is between 0 to 10 pixel or not, if not then next check 10 to 20.. so on 
#     for detection in result:
#         # Setting range (0-10) for first loop, then (10-20) for second loop and so on.
#         Range = range(flag, flag + C)
#         # check if text is with in range, if yes -> Then store those text in one list (i.e) these text which store in second for loop execution are one line.
#         if detection[0][0][1] in Range:
#             sub_list.append(detection[1])
#     # increment the range flag
#     flag += C
#     Main_list.append(sub_list)
#     # reset the line list and again store text for next line when for loop started.
#     sub_list = []
# # remove empty list in main list and make line by line list group. The data stored in format of list (main_list) in list (sub_list -> line data)
# Main_list = res = [ele for ele in Main_list if ele != []]

# ## Store the list of groups to each row of excel
# # open workbook
# wb = openpyxl.Workbook()
# sheet = wb.active
# p = 0
# # add row by row in excel from Main list
# for i in Main_list:
#     k = 0
#     for j in i:
#         c1 = sheet.cell(row=p + 1, column=k + 1)
#         c1.value = str(j)
#         k += 1
#     p += 1
# # save data in excel
# wb.save("output4.xlsx")

# import pandas as pd
# import easyocr
# import cv2
# from tabulate import tabulate


path = r'/home/samyak/table_data_extract/cropped_output_images/output1.jpg'

# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# noise=cv2.medianBlur(gray,3)
# thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# reader = easyocr.Reader(['en'])
# result = reader.readtext(img,paragraph='False')
# df=pd.DataFrame(result)
# # df.fillna("", inplace=True)
# print(tabulate(df[1]))
# # print(df[1])

import cv2
import pytesseract
import numpy as np
import pandas as pd
from tabulate import tabulate

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image, grayscale, Otsu's threshold
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Repair horizontal table lines 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55,2))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image, [c], -1, (255,255,255), 9)

# Remove vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,55))
detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image, [c], -1, (255,255,255), 9)

# Perform OCR
data = pytesseract.image_to_string(image, lang='eng',config='--psm 6')
# df = pd.DataFrame(columns=['data'])
print(data)

# cv2.imshow('image', image)
# cv2.imwrite('image7.png', image)
# cv2.waitKey()