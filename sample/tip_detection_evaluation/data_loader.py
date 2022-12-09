"""
It is a script used to load data
"""
import cv2
from os import listdir
from os.path import isdir, isfile, join

def load_images_from_dir(dir_path, image_type):
    """
    It checks the filename in directory. And return the jpeg images
    :param dir_path: the path of directory
    :return: jpeg images
    """
    images = []
    if isdir(dir_path):
        file_list = listdir(dir_path)
        file_list.sort(key= lambda x:int(x[:-4]))
        for file_obj in file_list:
            if isfile(join(dir_path, file_obj)):
                name, type_name = file_obj.split('.')
                if type_name == image_type:
                    images.append(file_obj)
                else:
                    print("It is not a {} file.".format(image_type))
            else:
                print("It is not a file.")
    else:
        print("The path should be a direcotry whcih contains the right files.")
    return images