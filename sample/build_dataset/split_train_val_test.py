"""
It is a program used to split dataset into train, val and test sets.
"""
from os import listdir
from os.path import isfile, join
import random

def main():
    print("start spliting dataset...")
    # configs
    dataset_path = "/home/raphael/Desktop/data/surgery/multi-label-seg/multiclass-voc/JPEGImages"
    image_set_path = "/home/raphael/Desktop/data/surgery/multi-label-seg/multiclass-voc/ImageSets/Segmentation"
    train_txt = "train.txt"
    val_txt = "val.txt"
    test_txt = "test.txt"


    # open files
    train = open(join(image_set_path, train_txt), 'a')
    val = open(join(image_set_path, val_txt), 'a')
    test = open(join(image_set_path, test_txt), 'a')

    train_num = 0
    val_num = 0
    test_num = 0
    # traverse the files
    for file_obj in listdir(dataset_path):
        if isfile(join(dataset_path, file_obj)):
            name = file_obj[:-4]
            subname = name.split('_')
            rand = random.randint(0, 9)
            if len(subname) > 0:
                if rand < 7:
                    train.write(name+'\n')
                    train_num += 1
                elif rand < 9:
                    val.write(name+'\n')
                    val_num += 1
                else:
                    test.write(name+'\n')
                    test_num += 1
            #else:
            #    test.write(name+'\n')
            #    test_num += 1
        print("train: " + str(train_num)
              + " val: " + str(val_num)
              + " test: " + str(test_num))

    # close files
    train.close()
    val.close()
    test.close()


if __name__ == "__main__":
    main()