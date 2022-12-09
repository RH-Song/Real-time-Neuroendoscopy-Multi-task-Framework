# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     maozezhong 2018-6-27
# Modified by Rihui Song, Hongli  2020.9.27
##############################################################

import time
import random
from cv2 import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def rotation(img,mask,rotation_angle = [0,90,180,270]):
    center = (round(img.shape[0]/2),round(img.shape[1]/2))
    idx = np.random.randint(0,4,dtype= np.int)
    angle = rotation_angle[idx]
    rot_mat =  cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    rot_mat =  cv2.getRotationMatrix2D(center, angle, 1)
    new_img_mask = cv2.warpAffine(mask, rot_mat, (img.shape[1], img.shape[0]))
    return new_img,new_img_mask

def changeLight(img):
    flag = random.uniform(0.5, 1.5)  
    return exposure.adjust_gamma(img, flag)

def addNoise(img):

    pro_noise = 0.2
    pro_specularity = 1
    if random.random()< pro_noise:
        img = random_noise(img, mode='gaussian', clip=True)*255
        img = img.astype(np.uint8)
    if random.random()< pro_specularity:
        img = add_specularity(img,max_num_specularity = 5,color = (150,150,150))
        return img


def filp_pic(img, mask):
    
    flip_img = img
    flip_mask = mask
    if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
        horizon = True
    else:
        horizon = False


    if horizon:  #水平翻转
        flip_img = cv2.flip(flip_img, 1)
        flip_mask = cv2.flip(flip_mask, 1)
    else:
        flip_img = cv2.flip(flip_img, 0)
        flip_mask = cv2.flip(flip_mask, 0)

    return flip_img, flip_mask

def add_specularity(img_raw,max_num_specularity = 8,color = (150,150,150)):
    img = img_raw
    row,col,dim = img.shape
    num_of_specularity = np.random.randint(1,max_num_specularity)
    #determine the number of specularity we add on the raw image
    for i in range(num_of_specularity):
        tmp_img = np.zeros(img.shape,dtype= np.uint8)
        long_axis =  np.random.randint(2,round(row/6),dtype= np.int)
        short_axis = np.random.randint(0,round(long_axis/2),dtype= np.int)
        center_x,center_y = np.random.randint(round(row*1/8),round(row*7/8)),np.random.randint(round(row*1/8),round(row*7/8),dtype= np.int)
        angle = np.random.randint(0,360,dtype= np.int)
        startAngle,endAngle = 0,360
        elp = cv2.ellipse(tmp_img,(center_x,center_y),(long_axis,short_axis),angle,startAngle,endAngle,color, thickness=-1)
        img = cv2.add(img,elp)
    return img

def dataAugment(img,mask):

    rotation_rate = 0.25
    change_light_rate = 0.25
    add_noise_rate = 1
    flip_rate = 0.5
    change_num = 0  #改变的次数
    print('------')
    while change_num < 1:   #默认至少有一种数据增强生效


        if random.random() > rotation_rate:    #旋转
            print('旋转')
            change_num += 1
            img, mask = rotation(img, mask)

        if random.random() > change_light_rate: #改变亮度
            print('亮度')
            change_num += 1
            img = changeLight(img)      

        if random.random() < add_noise_rate:    #加噪声/光斑
            print('加噪声')
            change_num += 1
            img = addNoise(img)

        if random.random() < flip_rate:    #翻转
            print('翻转')
            change_num += 1
            img, mask = filp_pic(img, mask)
        print('\n')

    return img, mask


if __name__ == '__main__':

    need_aug_num = 2  #一个文件需要增广到多少份

    source_img_root_path = 'D:/surgery_img/imgs/'
    source_mask_root_path = 'D:/surgery_img/masks/'
    img_output_dir = 'D:/surgery_img/output/imgs/'
    mask_output_dir = 'D:/surgery_img/output/masks/'

    cnt2 = 0
    for parent, _, files in os.walk(source_img_root_path):
        for file in files:
            cnt = 0
            while cnt < need_aug_num:
                print(file)
                pic_path = os.path.join(parent, file)
                mask_path = os.path.join(source_mask_root_path, file[:-4]+'.png')
            

                img = cv2.imread(pic_path)
                mask = cv2.imread(mask_path)
            
                cnt += 1
                cnt2 += 1
                auged_img,auged_mask = dataAugment(img,mask)
                cv2.imwrite(os.path.join(img_output_dir, file[:-4]+'_'+str(cnt)+'.jpg'), auged_img)
                cv2.imwrite(os.path.join(mask_output_dir, file[:-4]+'_'+str(cnt)+'.png'), auged_mask)
                print('Total number of finished images:')
                print(cnt2)