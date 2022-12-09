"""
It is a program convert png images to jpg images
"""
import cv2
from os import listdir
from os.path import join, isfile

def main():
    print('start converting png files to jpg files...')
    # configs
    dir_path = '/data/home/usi/data/surgery/1/'

    for file_obj in listdir(dir_path):
        if isfile(join(dir_path, file_obj)):
            name, extend_type = file_obj.split('.')
            if extend_type == "png":
                print(name)
                image = cv2.imread(join(dir_path, file_obj))
                cv2.imwrite(dir_path+name+'.jpg', image)

if __name__ == "__main__":
    main()