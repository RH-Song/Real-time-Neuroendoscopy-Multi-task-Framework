from os import listdir
from os.path import isfile, join
from shutil import copyfile

def main():
    print("start to pick and rename.")
    # configs
    root_path = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/json'
    new_path = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/integrated'

    for f in listdir(root_path):
        if isfile(join(root_path, f)):
            name, extend_type = f.split('.')
            if extend_type == 'json':
                image_src = join(root_path, name + '.json')
                image_target = join(new_path, name + "_b" + '.json')
                copyfile(image_src, image_target)

if __name__ == '__main__':
    main()