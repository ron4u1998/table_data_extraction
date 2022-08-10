# # import cv2
# # from TextTron.TextTron import TextTron

# # path = r'/home/samyak/table_data_extract/cropped_output_images/output1.jpg'

# # img = cv2.imread(path)
# # imS = cv2.resize(img, (960, 540))
# # cv2.imshow('image', imS)
# # cv2.waitKey(0)
# # # cv2.destroyAllWindows() 

# # TT = TextTron(img) 
# # TT = TextTron(img, low=196,high=255,yThreshold=15,xThreshold=4)
# # tbbox = TT.textBBox
# # plotImg = TT.plotImg
# # plot_img = cv2.resize(plotImg,(960,540))
# # cv2.imshow('plotted_image', plot_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

#importing libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the image
path = r'/home/samyak/table_data_extract/cropped_output_images/output1.jpg'
read_image = cv2.imread(path , 0)

#detecting the cells
#for purpose of converting an image to excel we need to first detect the cells that are horizontal and vertical lines from the cells

#first convert the image to BINARY and then turn them GRAYSCALE 
convert_bin, grey_scale = cv2.threshold(read_image, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
grey_scale = 255-grey_scale
grey_graph = plt.imshow(grey_scale, cmap= 'gray')
# plt.show()
