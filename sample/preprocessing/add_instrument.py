# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     Modified by  Hongli 2020.10.15
##############################################################
from cv2 import cv2
import numpy as np
import  os
import random

def add_instrument(img_raw,mask_raw,target_img_raw,target_mask_raw,rotation_angle,transparency = 0.75):
    """ 
    input：
    1. img_raw: raw image with instrument
    2. mask_raw: the mask of the raw image (upper)
    3. target_img_raw: the target image 
    4. target_mask_raw: the target mask  (lower)
    5. rotation_angle: perfer 90,180,270
    return:
    1. new_img: new image 
    2. new_mask: the mask of the new image
    """
    img = img_raw
    mask = mask_raw
    img_no_ins = target_img_raw
    new_img = np.zeros(img.shape,dtype=np.uint8)
    new_ins_mask = mask 
    center = (round(img.shape[0]/2),round(img.shape[1]/2))
    angle = rotation_angle
    idx = np.where(mask!=0)
    row = idx[0]
    col = idx[1]
    para = transparency  # adjust transparency
    ########## 
    # cut the instrument from raw image
    ##########
    for x in range(len(row)):
        new_img[row[x],col[x],0] = round(img[row[x],col[x],0]*para)
        new_img[row[x],col[x],1] = round(img[row[x],col[x],1]*para)
        new_img[row[x],col[x],2] = round(img[row[x],col[x],2]*para)

    rot_mat =  cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(new_img, rot_mat, (img.shape[1], img.shape[0]))
    rot_mat =  cv2.getRotationMatrix2D(center, angle, 1)
    new_ins_mask = cv2.warpAffine(new_ins_mask, rot_mat, (img.shape[1], img.shape[0]))
    ########## 
    # add the instrument to the target img
    ##########
    idx = np.where(new_ins_mask!=0)
    row = idx[0]
    col = idx[1]
    for x in range(np.size(row)):
        img_no_ins[row[x],col[x],0] = round(img_no_ins[row[x],col[x],0]*(1-para))
        img_no_ins[row[x],col[x],1] = round(img_no_ins[row[x],col[x],1]*(1-para))
        img_no_ins[row[x],col[x],2] = round(img_no_ins[row[x],col[x],2]*(1-para))    
    new_img = cv2.add(new_img,img_no_ins)
    ########## 
    # add the raw mask to the target mask
    ##########
    cmap = label_colormap()
    new_mask = target_mask_raw
    num_color, color_set = num_of_color(new_ins_mask)
    target_num_color, target_color_set = num_of_color(new_mask)

    # renumber colors of target mask (Incase the target mask has different order of colors)
    for i,vector in enumerate(target_color_set):
        (px0,px1,px2) = vector
        idx = np.where((new_mask[:,:,0] == px0)&(new_mask[:,:,1] == px1)&(new_mask[:,:,2] == px2))
        row = idx[0]
        col = idx[1]
        for x in range(np.size(row)):
            new_mask[row[x],col[x],:] = cmap[i+1,:]

    for i,vector in enumerate(color_set):
        (px0,px1,px2) = vector
        idx = np.where((new_ins_mask[:,:,0] == px0)&(new_ins_mask[:,:,1] == px1)&(new_ins_mask[:,:,2] == px2))
        row = idx[0]
        col = idx[1]
        for x in range(np.size(row)):
            new_mask[row[x],col[x],:] = cmap[i+target_num_color+1,:]
   
    
    return new_img,new_mask

def num_of_color(img):
    n = 0
    vector_set = set()

    for x in  range(img.shape[0]):
        for y in range(img.shape[1]):
            px_0 = img[x,y,0]
            px_1 = img[x,y,1]
            px_2 = img[x,y,2]
            tmp = (px_0,px_1,px_2)
            vector_set.add(tmp)
    
    vector_set.remove((0,0,0))
    n = len(vector_set) # non-while color number
    return n, vector_set


def rot_overlap(img_raw,img_overlap,mask_raw,mask_overlap):
    """ Cover the instrument in a rotating way """
    gray = cv2.cvtColor(mask_raw, cv2.COLOR_RGB2GRAY)
    ret, binary_raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    gray = cv2.cvtColor(mask_overlap, cv2.COLOR_RGB2GRAY)
    ret, binary_overlap = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    nonzero = cv2.findNonZero(binary_raw)
    for cord in nonzero:
        tmp = cord[0]
        img_raw[tmp[1],tmp[0],:] = img_overlap[tmp[1],tmp[0],:]
        mask_raw[tmp[1],tmp[0],:] = mask_overlap[tmp[1],tmp[0],:]
    return img_raw,mask_raw

def rotation(img,angle):
    center = (round(img.shape[0]/2),round(img.shape[1]/2))
    rot_mat =  cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    
    return new_img

def translation_overlap(img,mask,offset = 100):
    """ Cover the instrument in a translating way """
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    nonzero = cv2.findNonZero(binary) 
    for cord in nonzero:
        tmp = cord[0]
        img[tmp[1],tmp[0],:] = img[tmp[1]-offset,tmp[0]-offset,:]
        mask[tmp[1],tmp[0],:] = mask[tmp[1]-offset,tmp[0]-offset,:]
    return img,mask

def targer_img_process(target_img,target_mask):
    """ 
    Input the target image and mask
    return the image and mask with covered instrumment
    """
    img = target_img
    mask = target_mask
    num_of_mask_pixel = 1
    cnt = 1
    while(num_of_mask_pixel != 0):
        cnt += 1
        img_rot = rotation(img,90)
        mask_rot = rotation(mask,90)
        img, mask = rot_overlap(img,img_rot,mask,mask_rot)

        idx = np.where(mask != 0)
        num_of_mask_pixel = len(idx[0])
        if cnt>8:
            img,mask = translation_overlap(img,mask)
            break
    return img,mask


def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    # if value is not None:
    #     hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
    #     if isinstance(value, float):
    #         hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
    #     else:
    #         assert isinstance(value, int)
    #         hsv[:, 1:, 2] = value
    #     cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
    return cmap

def image_augmentation(source_image_root_path, source_mask_root_path,
                       target_pic_dir, targer_mask_dir,
                       image_store_dir, marsk_store_dir):
    prob = 0.5
    cnt = 0
    num = 1  # Cycles
    for x in range(num):
        for parent, _, files in os.walk(
                source_image_root_path):  # For every instrument, we randomly choose a target image
            for file in files:
                cnt += 1
                pic_path = os.path.join(parent, file)
                mask_path = os.path.join(source_mask_root_path, file[:-4] + '.png')

                # Random selection of target image
                dirs = os.listdir(target_pic_dir)
                length = len(dirs)
                idx = np.random.randint(0, length, dtype=np.int)
                target_path_img = os.path.join(target_pic_dir, dirs[idx])
                target_path_mask = os.path.join(targer_mask_dir, dirs[idx])
                img = cv2.imread(pic_path)
                mask = cv2.imread(mask_path)
                target_img = cv2.imread(target_path_img)
                target_mask = cv2.imread(target_path_mask[:-4] + '.png')

                # random angle for the instrument mapping
                angle = [0, 90, 180, 270]
                idx = np.random.randint(0, len(angle), dtype=np.int)

                if random.random() > prob:
                    print('Mode1')  # The original instrument is covered by the background
                    target_img, target_mask = targer_img_process(target_img, target_mask)
                    new_img, new_mask = add_instrument(img, mask, target_img, target_mask, angle[idx], transparency=0.7)
                else:
                    print('Mode2')  # The original instrument is not covered
                    new_img, new_mask = add_instrument(img, mask, target_img, target_mask, angle[idx], transparency=1.0)

                cv2.imwrite(image_store_dir + file, new_img)
                cv2.imwrite(marsk_store_dir + file[:-4] + '.png', new_mask)
                print('Number of finished images:')
                print(cnt)

if __name__ == '__main__':

    source_image_root_path = 'D:/surgery_img/all/a/imgs/'    
    source_mask_root_path = 'D:/surgery_img/all/a/masks/'    
    target_pic_dir = 'D:/surgery_img/tar/a/imgs/'              
    targer_mask_dir = 'D:/surgery_img/tar/a/masks/'
    image_store_dir = 'D:/surgery_img/result/imgs/'
    marsk_store_dir = 'D:/surgery_img/result/masks/'

    image_augmentation(source_image_root_path, source_mask_root_path,
                       target_pic_dir, targer_mask_dir,
                       image_store_dir, marsk_store_dir)