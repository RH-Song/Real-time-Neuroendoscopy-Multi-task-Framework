import cv2
import os
import numpy as np
from PIL import Image

if __name__ == '__main__':
    source_pic_root_path = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc/JPEGImages'
    source_annotation_root_path = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc/SegmentationClass'
    target_pic_dir = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc_aug/JPEGImages'
    target_annotation_dir = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images/hand_voc_aug/SegmentationClass'

    #src_img_name = target_anno[:-6]+'.jpg'
    #print(src_img_name)
    #src_anno_img_name = src_img_name[:-4]+'.png'
    #src_img = cv2.imread(os.path.join(source_pic_root_path, src_img_name))
    #print('src img: {}'.format(src_img.shape))
    #src_anno_img = cv2.imread(os.path.join(source_annotation_root_path, src_anno_img_name))
    #print(src_anno_img.shape)
    #print(np.unique(src_anno_img.reshape(-1, src_anno_img.shape[-1]), axis=0, return_counts=True))
    target_annotations = os.listdir(target_annotation_dir)
    target_anno = target_annotations[0]
    target_anno_img = cv2.imread(os.path.join(target_annotation_dir, target_anno))
    pixel_type, num = np.unique(target_anno_img.reshape(-1, target_anno_img.shape[-1]), axis=0, return_counts=True)
    print(pixel_type)
    print(num)
    ##target_anno_img = Image.open(os.path.join(target_annotation_dir, target_anno))
    ##src_annotations = os.listdir(source_annotation_root_path)
    #src_anno = src_annotations[0]
    #src_anno_img = cv2.imread(os.path.join(source_annotation_root_path, src_anno))

    for target_anno in target_annotations:
        #target_img_name = target_anno[:-4]+'.jpg'

        #tar_img = cv2.imread(os.path.join(target_pic_dir, target_img_name))
        #cv2.imwrite(os.path.join(target_pic_dir, 'aug'+str(index)+'.jpg'), tar_img)
        #if tar_img.shape != (480,640,3):
        #    print('target img: {}'.format(index))

        print(target_anno)
        target_anno_img = Image.open(os.path.join(target_annotation_dir, target_anno))
        w, h = target_anno_img.size
        for w_i in range(0,w):
            for h_i in range(0,h):
                if target_anno_img.getpixel((w_i, h_i)) != 0:
                    target_anno_img.putpixel((w_i, h_i), 128)
        #p_anno_img = target_anno_img.convert('P')
        target_anno_img.save(os.path.join(target_annotation_dir, target_anno))
        #cv2.imwrite(os.path.join(target_annotation_dir, 'aug'+str(index)+'.png'), target_anno_img)
        #if target_anno_img.shape != (480,640,3):
        #    print('target anno img: {}'.format(index))
        #    print(target_anno_img.shape)
        #pixel_type, _ = np.unique(target_anno_img.reshape(-1, target_anno_img.shape[-1]), axis=0, return_counts=True)
        #if len(pixel_type) != 2:
        #    print(target_anno)
        #index += 1
