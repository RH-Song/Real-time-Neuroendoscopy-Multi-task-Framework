# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for Segmentation
# author:
#     maozezhong 2018-6-27
# Modified by Rihui Song 2019-11-28
##############################################################

# 包括:
#     1. 裁剪(需改变mask)
#     2. 平移(需改变mask)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变mask)
#     6. 镜像(需要改变mask)
#     7. cutout
# 注意:   
#     random.seed(),相同的seed,产生的随机数是一样的!!


import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def show_pic(img, anno_img=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.namedWindow('anno_img', 0)  # 1表示原图
    cv2.moveWindow('anno_img', 0, 0)
    cv2.resizeWindow('anno_img', 1200,800)  # 可视化的图片大小
    cv2.imshow('anno_img', anno_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')



# 图像均为cv2读取
class DataAugmentForSegmentaion():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=180,
                 crop_rate=0.5, shift_rate=0.75, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.75):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    # calculate bbox for segmentation part
    def _calculate_bbox_for_seg(self, anno_img):
        # countour for hand
        gray_image = cv2.cvtColor(anno_img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(cnt)
        bbox = [int(mask_x - mask_h/2), int(mask_y - mask_w/2), int(mask_x + mask_h/2), int(mask_y + mask_w/2)]
        return bbox

    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time())) 
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True)*255

    
    # 调整亮度
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)
    
    # cutout
    def _cutout(self, img, anno_img, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        
        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            
            chongdie = True    #看切割的区域是否与box重叠太多
            
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)    #numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                bbox = self._calculate_bbox_for_seg(anno_img)
                if cal_iou([x1,y1,x2,y2], bbox) > threshold:
                    chongdie = True
                    break
            
            mask[y1: y2, x1: x2, :] = 0.
        
        # mask = np.expand_dims(mask, axis=0)
        img = img * mask
        anno_img = anno_img * mask

        return img, anno_img

    # 旋转
    def _rotate_img_anno(self, img, anno_img, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            anno_img: mask for img
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_anno_img: mask after rotation
        '''
        #---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        rot_anno_img = cv2.warpAffine(anno_img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        return rot_img, rot_anno_img


    # 裁剪
    def _crop_img_anno(self, img, anno_img):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            anno_img: annotation image
        输出:
            crop_img:裁剪后的图像array
            crop_anno_img:裁剪后的annotation image
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]

        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0

        bbox = self._calculate_bbox_for_seg(anno_img)

        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
        d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离

        #随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        crop_anno_img = anno_img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        return crop_img, crop_anno_img
    # 平移
    def _shift_pic_anno(self, img, anno_img):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        #---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        bbox = self._calculate_bbox_for_seg(anno_img)
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最大左移动距离
        d_to_right = w - x_max      #包含所有目标框的最大右移动距离
        d_to_top = y_min            #包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max     #包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])  #x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        shift_anno_img = cv2.warpAffine(anno_img, M, (img.shape[1], img.shape[0]))

        return shift_img, shift_anno_img

    # 镜像
    def _filp_pic_anno(self, img, anno_img):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        flip_anno_img = copy.deepcopy(anno_img)
        # if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
        horizon = True
        # else:
            # horizon = False
        h, w, _ = img.shape
        if horizon:  #水平翻转
            flip_img = cv2.flip(flip_img, 1)
            flip_anno_img = cv2.flip(flip_anno_img, 1)
        else:
            flip_img = cv2.flip(flip_img, 0)
            flip_anno_img = cv2.flip(flip_anno_img, 0)

        return flip_img, flip_anno_img

    def dataAugment(self, img, anno_img):
        '''
        图像增强
        输入:
            img:图像array
            anno_img: 分割标签array
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的分割标签array
        '''
        change_num = 0  #改变的次数
        print('------')
        while change_num < 1:   #默认至少有一种数据增强生效

            #if random.random() < self.crop_rate:        #裁剪
            #    print('裁剪')
            #    change_num += 1
            #    img, anno_img= self._crop_img_anno(img, anno_img)

            if random.random() > self.rotation_rate:    #旋转
                print('旋转')
                change_num += 1
                angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                #angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.7, 0.8)
                img, anno_img = self._rotate_img_anno(img, anno_img, angle, scale)
            
            if random.random() < self.shift_rate:        #平移
                print('平移')
                change_num += 1
                img, anno_img = self._shift_pic_anno(img, anno_img)
            
            if random.random() > self.change_light_rate: #改变亮度
                print('亮度')
                change_num += 1
                img = self._changeLight(img)
            
            #if random.random() < self.add_noise_rate:    #加噪声
            #    print('加噪声')
            #    change_num += 1
            #    img = self._addNoise(img)

            if random.random() < self.cutout_rate:  #cutout
                print('cutout')
                change_num += 1
                img, anno_img = self._cutout(img, anno_img, length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:    #翻转
                print('翻转')
                change_num += 1
                img, anno_img = self._filp_pic_anno(img, anno_img)
            print('\n')
        # print('------')
        return img, anno_img

if __name__ == '__main__':

    print('Start Segmentation')
    import shutil

    need_aug_num = 3

    dataAug = DataAugmentForSegmentaion()

    source_pic_root_path = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc/JPEGImages'
    source_annotation_root_path = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc/SegmentationClass'
    target_pic_dir = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc_aug/JPEGImages'
    target_annotation_dir = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc_aug/SegmentationClass'

    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:
            cnt = 0
            while cnt < need_aug_num:
                print(file)
                pic_path = os.path.join(parent, file)
                annotation_path = os.path.join(source_annotation_root_path, file[:-4]+'.png')

                img = cv2.imread(pic_path)
                anno_img = cv2.imread(annotation_path)

                auged_img, auged_annotation = dataAug.dataAugment(img, anno_img)
                resize_auged_img = cv2.resize(auged_img, (640,480))
                resize_anged_annotation = cv2.resize(auged_annotation, (640,480))
                w,h,d = resize_anged_annotation.shape
                for w_i in range(0,w):
                    for h_i in range(0,h):
                        if resize_anged_annotation[w_i][h_i][2] != 128:
                            resize_anged_annotation[w_i][h_i][2] = 0
                cnt += 1

                # show_pic(auged_img, auged_annotation)  # 强化后的图
                # save arged images and annotations
                cv2.imwrite(os.path.join(target_pic_dir, file[:-4]+'.'+str(cnt)+'.jpg'), resize_auged_img)
                cv2.imwrite(os.path.join(target_annotation_dir, file[:-4]+'.'+str(cnt)+'.png'), resize_anged_annotation)

