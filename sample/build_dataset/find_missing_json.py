from os import listdir
from os.path import isfile, join

def main():
    print("start finding...")
    # configs
    root_path = '/home/raphael/Desktop/data/surgery/single-label-seg/allmask/origin_annotation/all_single_imgs'

    for f in listdir(root_path):
       if isfile(join(root_path, f)):
           name, extend_type = f.split('.')
           if extend_type == 'json':
               json_src = join(root_path, name + '.jpg')
               if not isfile(join(root_path, json_src)):
                   print(name)

if __name__ == '__main__':
    main()
