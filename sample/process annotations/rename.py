from os import listdir
from os.path import join
import cv2

if __name__ == '__main__':
    origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/ground_truth"
    dir_path = origin_path + '/mask'
    src_path = origin_path + '/new_mask'

    file_objs = listdir(dir_path)
    for file_obj in file_objs:
        img = cv2.imread(join(dir_path, file_obj))
        name, _ = file_obj.split('_')
        cv2.imwrite(join(src_path, name+'.png'), img)
