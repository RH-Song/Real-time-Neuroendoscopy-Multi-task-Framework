"""
It is a program that used to parse xml annotations to get labels and write them into a txt file
"""

from os import listdir
from os.path import join, isfile
import json

def main():
    # configs
    old_annotation_dir = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style/json_no_imagedata"
    new_annotation_dir = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style/json"

    for annotation_obj in listdir(old_annotation_dir):
        if isfile(join(old_annotation_dir, annotation_obj)):
            name, extended_name = annotation_obj.split('.')
            print(name)
            if extended_name == "json":
                with open(join(old_annotation_dir, annotation_obj)) as src_f:
                    annotation_dict = json.load(src_f)
                    annotation_dict["imageData"] = None
                    json.dump(annotation_dict, open(join(new_annotation_dir, name+"_a"+".json"), 'w'), indent=4)

if __name__ == '__main__':
    print("Add element 2 json.")
    main()