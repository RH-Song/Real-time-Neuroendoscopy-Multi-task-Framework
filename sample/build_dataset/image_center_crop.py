"""
It is a program used to crop the center part of the image.
And it will adjust the annotation of the image at the same time.
"""
from os import listdir
from os.path import join, isfile
import cv2
from xml.dom import minidom

def shift_x(x, width, height):
    delta_x = int(width - height) / 2
    x = x - delta_x
    if x < 0:
        x = 0
    elif x >= height:
        x = height - 1
    return int(x)

def main():
    print('start center cropping...')
    # configs
    images_dir = '/home/raphael/Desktop/data/surgery/single-label-seg/hard_cases/specular0'
    annotation_dir = '/data/home/usi/data/surgery/surgery_VOCstyle/Annotations'
    new_images_dir = '/home/raphael/Desktop/data/surgery/single-label-seg/hard_cases/specular0'
    new_annotation_dir = '/data/home/usi/data/surgery/squared/Annotations'

    crop_xml = False

    for image_obj in listdir(images_dir):
        # crop image
        print(image_obj)
        image = cv2.imread(join(images_dir, image_obj))
        height, width, channel = image.shape
        width_start = int((width - height) / 2)
        width_end = int(width_start + height)
        cropped_image = image[:,width_start:width_end,:]
        cv2.imwrite(join(new_images_dir, image_obj), cropped_image)

        if crop_xml:
            # parse the xml file
            name, _ = image_obj.split('.')
            xml_file_name = name + '.xml'
            print(xml_file_name)
            xml = minidom.parse(join(annotation_dir, xml_file_name))
            root = xml.documentElement
            # change the size of image
            xml_widths = root.getElementsByTagName('width')
            xml_width = xml_widths[0]
            xml_width.firstChild.data = height

            # change the xmin
            xml_xmins = root.getElementsByTagName('xmin')
            for xmin in xml_xmins:
                xmin.firstChild.data = shift_x(int(xmin.firstChild.data), width, height)

            # change the xmax
            xml_xmaxs = root.getElementsByTagName('xmax')
            for xmax in xml_xmaxs:
                xmax.firstChild.data = shift_x(int(xmax.firstChild.data), width, height)

            with open(join(new_annotation_dir, xml_file_name), 'w') as fh:
                xml.writexml(fh)

if __name__ == '__main__':
    main()