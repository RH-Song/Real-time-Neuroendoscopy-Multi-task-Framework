"""
It is a program that used to parse xml annotations to get labels and write them into a txt file
"""

from os import listdir
from os.path import join, isfile
import json

def checkLabel(labels, label):
    for l in labels:
        if l == label:
            return False
    return True

def main():
    # configs
    annotation_dir = "/home/raphael/Desktop/data/surgery/multi-label-seg/origin_dataset/annotations"
    txt_path = "/home/raphael/Desktop/data/surgery/a/tracking/02-02-26/02-02-26.txt"

    labels = []

    for annotation_obj in listdir(annotation_dir):
        if isfile(join(annotation_dir, annotation_obj)):
            name, extended_name = annotation_obj.split('.')
            print(name)
            if extended_name == "json":
                with open(join(annotation_dir, annotation_obj)) as src_f:
                    annotation_dict = json.load(src_f)
                    shapes = annotation_dict["shapes"]
                    for shape in shapes:
                        label = shape["label"]
                        if checkLabel(labels, label):
                            labels.append(label)

    print(labels)
    # txt = open(txt_path, 'w')
    #
    # txt.close()

if __name__ == '__main__':
    print("get labels from annotations.")
    main()