"""
It is a program used to pick images randomly from one directory to another.
Author: Rihui Song
"""
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import random

def main():
    print("start to pick images.")
    # configs
    dir_path = '/home/raphael/Desktop/data/surgery/wcd/images'
    dir_save_images = '/home/raphael/Desktop/data/surgery/wcd/500'

    # traverse the files
    for f in listdir(dir_path):
        if isfile(join(dir_path, f)):
            rand = random.randint(1, 8)
            if rand <= 5:
                image_src = join(dir_path, f)
                image_target = join(dir_save_images, f)
                copyfile(image_src, image_target)

if __name__ == "__main__":
    main()
