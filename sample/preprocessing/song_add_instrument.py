# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     Modified by  song 2020.10.16
##############################################################

import add_instrument

if __name__ == '__main__':

    source_image_root_path = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/seg_coco_style/train_voc/JPEGImages/'
    source_mask_root_path = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/seg_coco_style/train_voc/SegmentationObjectPNG/'
    
    store_pic_dir = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/imgs/'
    store_mask_dir = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/masks/'

    add_instrument.image_augmentation(source_image_root_path, source_mask_root_path,
                                      source_image_root_path, source_mask_root_path,
                                      store_pic_dir, store_mask_dir)