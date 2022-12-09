"""
It is a program used to pick images from one directory and change to binary images.
Author: Rihui Song
"""
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import random
import cv2

def main():
    print("start to pick images.")
    # configs
    dir_path = '/home/raphael/Desktop/data/surgery/surgery_segmentation_backup/surgery_seg_dataset/instance_ground_true'
    dir_save_images = '/home/raphael/Desktop/data/surgery/surgery_segmentation_backup/surgery_seg_dataset/semantic_gt'

    count = 0
    # traverse the files
    for f in listdir(dir_path):
        count += 1
        src_img_path = join(dir_path, f)
        target_img_path = join(dir_save_images,f)
        if isfile(src_img_path):
            src_img = cv2.imread(src_img_path)
            h = src_img.shape[0]
            w = src_img.shape[1]
            for r in range(h):
                for c in range(w):
                    if (src_img[r][c] != [0,0,0]).any():
                        src_img[r][c] = (255,255,255)
            # cv2.imshow("test", src_img)
            # cv2.waitKey(0)
            cv2.imwrite(target_img_path, src_img)
            print(count)

if __name__ == "__main__":
    main()
