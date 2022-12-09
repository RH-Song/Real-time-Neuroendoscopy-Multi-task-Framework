"""
Split image according to txt file
"""

from os.path import join
from shutil import copyfile

if __name__ == '__main__':
    print('start split.')
    dir_path = '/home/raphael/Desktop/data/surgery/surgery_seg_dataset'
    txt_file = 'val.txt'
    img_dir = dir_path + '/img'
    val_dir = dir_path + '/val'

    val_txt = open(join(dir_path, txt_file), 'r')
    lines = val_txt.readlines()
    for line in lines:
        filename = line.split('\n')
        print('filename')
        print(filename[0] + '.jpg')
        image_src = join(img_dir, filename[0]+'.jpg')
        image_target = join(val_dir, filename[0]+'.jpg')
        copyfile(image_src, image_target)
