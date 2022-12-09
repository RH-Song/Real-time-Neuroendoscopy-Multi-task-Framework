"""
It is a program used to rename the label of objects.
"""
from os import listdir
from os.path import isfile, join
import json
from shutil import copyfile

def main():
    print("Start refactor labels of annotations.")

    # config
    src_dir = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/origin_annotation/H"
    dst_dir = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/refacted_annotation/G"

    for annotation_obj in listdir(src_dir):
        if isfile(join(src_dir, annotation_obj)):
            name, extended_name = annotation_obj.split('.')
            print(name)
            if extended_name == "json":
                with open(join(src_dir, annotation_obj)) as src_f:
                    annotation_dict = json.load(src_f)
                    shapes = annotation_dict["shapes"]
                    for shape in shapes:
                        shape["label"] = "tool"
                    with open(join(dst_dir, annotation_obj), 'w') as dst_f:
                        json.dump(annotation_dict, dst_f, indent = 4)

                    f = name+".jpg"
                    image_src = join(src_dir, f)
                    image_target = join(dst_dir, f)
                    copyfile(image_src, image_target)

if __name__ == '__main__':
    main()