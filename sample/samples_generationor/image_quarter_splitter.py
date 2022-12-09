"""
It is a program used to split an image into four parts. Each is a quarter of the image.
"""
from os import listdir
from os.path import isfile, join
import cv2

def quarter_split(img):
    """
    It is a function used to split the input image into four parts.
    :param img: an image
    :return: [four parts of the image]
    """
    hight, width, depth = img.shape

    half_hight = int(hight / 2)
    half_width = int(width / 2)

    parts = []
    parts.append(img[0:half_hight, 0:half_width, :])
    parts.append(img[half_hight:, 0:half_width, :])
    parts.append(img[0:half_hight, half_width:, :])
    parts.append(img[half_hight:, half_width:, :])

    return parts

def main():
    print("Start splitting...")

    # configs
    image_dir = '/home/raphael/Desktop/data/surgery/tool classes/temp'
    image_des = '/home/raphael/Desktop/data/surgery/tool classes/parts'

    # tranvers image dir
    for image_obj in listdir(image_dir):
        name, extend = image_obj.split('.')
        image_path = join(image_dir, image_obj)
        if isfile(image_path):
            origin_image = cv2.imread(image_path)
            parts = quarter_split(origin_image)
            # visualization
            # cv2.imshow('1', parts[0])
            # cv2.imshow('2', parts[1])
            # cv2.imshow('3', parts[2])
            # cv2.imshow('4', parts[3])
            # cv2.waitKey(0)
            cv2.imwrite(join(image_des, name+'_00.jpg'), parts[0])
            cv2.imwrite(join(image_des, name+'_10.jpg'), parts[1])
            cv2.imwrite(join(image_des, name+'_01.jpg'), parts[2])
            cv2.imwrite(join(image_des, name+'_11.jpg'), parts[3])

if __name__ == '__main__':
    main()