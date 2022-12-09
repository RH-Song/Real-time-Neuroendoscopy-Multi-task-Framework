"""
It is a script used to find out file in dir A but not in dir B.
"""
from os import listdir
from os.path import isfile, join

def main():
    print("start compare...")

    # config
    dir_with_more_files = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/voc_seg_style/integrated/imgs"
    dir_with_less_files = "/home/raphael/Desktop/data/surgery/surgery_segmentation_backup/surgery_seg_dataset/instance_ground_true"

    for file_obj in listdir(dir_with_more_files):
        file_name = file_obj[:-4]
        file_obj_json = file_name+"_json.png"
        if isfile(join(dir_with_less_files, file_obj_json)):
            pass
        else:
            print(file_name)

if __name__ == "__main__":
    main()