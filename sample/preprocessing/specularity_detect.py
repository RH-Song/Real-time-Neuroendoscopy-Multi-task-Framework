# -*- coding=utf-8 -*-
##############################################################
# description:
#     specularity detection 
# author:
#     Modified by  Hongli 2020.10.2
##############################################################
from cv2 import cv2
import numpy as np
import os


def specularity_det(img,mask,threshold,nu = 0.5):
    """ 
    Input: 
    1. image and the mask
    2. threshold: a 0-1 value
    Return:
    1. percentage: the specularity pixel on he instrument/ instrument pixel
    2. flag: whether the percentage is under the threshold
    (Ture: percentage > threshold)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray/255
    height, width, dim = img.shape
    canny = cv2.Canny(img,20,80)

    Img3c = img.reshape(height* width,3)  #verctorize
    Imin = np.min(Img3c,axis=1)  
    Imax =np.max(Img3c, axis=1)
    Ithresh = np.mean(Imin) + nu * np.std(Imin)
    tmp = Imin
    for x in range(len(Imin)):
        if Imin[x] > Ithresh:
            tmp[x] = 1
        else:
            tmp[x] = 0
    a = np.tile(Imin*tmp, (3,1))
    b = np.tile(Ithresh*tmp,(3,1))
    Iss = Img3c - a.T + b.T
    IBeta = (Imin - Ithresh)*tmp
    IBeta = np.uint8(IBeta)
    IHighlight = IBeta.reshape(height,width,1)
    ret, binary = cv2.threshold(IHighlight, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    image , contours , hierarchy = cv2.findContours (binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )

    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    nonzero = cv2.findNonZero(binary_mask)

    for cord in nonzero:
        tmp = cord[0]
        binary_mask[tmp[1],tmp[0]] = image[tmp[1],tmp[0]]

    nonzero_new = cv2.findNonZero(binary_mask)
    if nonzero_new is not None:
        percentage = len(nonzero_new)/len(nonzero)
    else:
        percentage = 0

    if percentage > threshold:
        flag = True
    else:
        flag = False
    """ 
    cv2.imshow('image',image)
    cv2.imshow('binary_mask',binary_mask)
    cv2.waitKey(0) 
    """ 
    #print(str(flag))


    return percentage,flag


if __name__ == '__main__':
    path_img = 'D:/surgery_img/imgs/'
    path_mask = 'D:/surgery_img/masks/'
    with open('D:/surgery_img/img_specularity.txt', 'w') as f:     

        # Threshold
        f.write('Threshold = ')
        f.write(str(0.1))
        for parent, _, files in os.walk(path_img):  #For every instrument, we randomly choose a target image 
                for file in files:
                    pic_path = os.path.join(parent, file)
                    img = cv2.imread(pic_path)

                    mask_path = os.path.join(path_mask, file[:-4]+'.png')
                    mask = cv2.imread(mask_path)
                    #image = fix_image_size(image)
                    percentage,flag = specularity_det(img,mask, threshold=0.1)   
                    print('The percentage is:',percentage)

                    ##
                    f.write('\n')
                    f.write(file+':  ')
                    f.write(str(flag))   
                    f.write('   Score =  ')
                    f.write(str(percentage))         
    f.close()



#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
#dilated = cv2.dilate(image,kernel)
#cv2.imshow('dilated',dilated)
