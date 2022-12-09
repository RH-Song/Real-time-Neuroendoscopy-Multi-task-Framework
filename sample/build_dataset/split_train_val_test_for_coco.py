"""
It is a program used to split dataset into train, val and test sets.
"""
from os import listdir, mkdir
from os.path import isfile, join, isdir
import random
from shutil import copyfile

def main():
    print("start spliting dataset...")
    # configs
    dataset_path = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/seg_coco_style/"
    image_path = dataset_path + "images"
    annotations_path = dataset_path + "annotations"

    train_path = dataset_path + "train"
    if not isdir(train_path):
        mkdir(train_path)
    val_path = dataset_path + "val"
    if not isdir(val_path):
        mkdir(val_path)
    test_path = dataset_path + "test"
    if not isdir(test_path):
        mkdir(test_path)

    train_num = 0
    val_num = 0
    test_num = 0
    # traverse the files
    for file_obj in listdir(image_path):
        if isfile(join(image_path, file_obj)):
            name, _ = file_obj.split('.')
            rand = random.randint(0, 9)
            if rand < 7:
                copyfile(join(image_path, file_obj), join(train_path, file_obj))
                copyfile(join(annotations_path, name+".json"), join(train_path, name+".json"))
                train_num += 1
            elif rand == 7:
                copyfile(join(image_path, file_obj), join(val_path, file_obj))
                copyfile(join(annotations_path, name+".json"), join(val_path, name+".json"))
                val_num += 1
            else:
                copyfile(join(image_path, file_obj), join(test_path, file_obj))
                copyfile(join(annotations_path, name+".json"), join(test_path, name+".json"))
                test_num += 1
        print("train: " + str(train_num)
              + " val: " + str(val_num)
              + " test: " + str(test_num))


if __name__ == "__main__":
    main()