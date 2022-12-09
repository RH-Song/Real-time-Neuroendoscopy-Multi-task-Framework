# -*- coding=utf-8 -*-
##############################################################
# description:
#     blur detection 
# author:
#     Modified by  Hongli 2020.10.2
##############################################################

from cv2 import cv2
import numpy 
import os

def fix_image_size(image: numpy.array, expected_pixels: float = 1E5):
    ratio = expected_pixels / (image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: numpy.array, mask,threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    _ , contours , hierarchy = cv2.findContours (binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    score = 1e5 # a large number 
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        ins_img = image[y:y+h,x:x+w]
        """ 
        cv2.namedWindow('ins_img',cv2.WINDOW_NORMAL)
        cv2.imshow('ins_img',ins_img)
        cv2.waitKey(0)
        ins_img = fix_image_size(ins_img)
        """
        blur_map = cv2.Laplacian(ins_img, cv2.CV_64F)
        if numpy.var(blur_map) != 0 :
            score = min(numpy.var(blur_map),score)

    return  score, bool(score < threshold)


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


if __name__ == '__main__':
    path_img = 'D:/surgery_img/imgs/'
    path_mask = 'D:/surgery_img/masks/'
    results = []
    with open('D:/surgery_img/img_blur.txt', 'w') as f:     # 打开test.txt   如果文件不存在，创建该文件。
        f.write('Threshold = ')
        f.write(str(100))
        for parent, _, files in os.walk(path_img):  #For every instrument, we randomly choose a target image 
                for file in files:
                    pic_path = os.path.join(parent, file)
                    img = cv2.imread(pic_path)

                    mask_path = os.path.join(path_mask, file[:-4]+'.png')
                    mask = cv2.imread(mask_path)

                    score, blurry = estimate_blur(img,mask, threshold=100)   
                    print('The score is:',score)
                    results.append(blurry)

                    ##
                    f.write('\n')
                    f.write(file+':  ')
                    f.write(str(blurry))   
                    f.write('   Score =  ')
                    f.write(str(score))         
    f.close()


"""
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',image)
cv2.namedWindow('result',cv2.WINDOW_NORMAL)
cv2.imshow('result', pretty_blur_map(blur_map))
cv2.waitKey(0) 
"""