"""
It is a scripts used to add masks to images to visualize the results.
"""
from os import listdir
from os.path import isfile, join
import cv2

def main():
    print("start visualize ...")
    # configs
    jpg_path = "/home/raphael/Desktop/data/surgery/surgery_segmentation_backup/brain_surgeries/a/tracking/02-02-26/JPEGImages"
    png_path = "/home/raphael/Desktop/data/surgery/surgery_segmentation_backup/brain_surgeries/a/tracking/02-02-26/results"
    dst_path = "/home/raphael/Desktop/data/surgery/surgery_segmentation_backup/brain_surgeries/a/tracking/02-02-26/v_results"

    for file_obj in listdir(png_path):
        mask = join(png_path, file_obj)
        if isfile(mask):
            name = file_obj[:-4]
            img = join(jpg_path, name+'.jpg')
            src1 = cv2.imread(img, cv2.IMREAD_COLOR)
            src2 = cv2.imread(mask, cv2.IMREAD_COLOR)

            dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
            # cv2.imshow('dst', dst)
            # cv2.waitKey(0)
            cv2.imwrite(join(dst_path, name+'.png'), dst)

if __name__ == "__main__":
    main()