"""
It is a program pick the images which have an xml file with the same name.
And the images will be rename as the order they were picked.
"""
from os import listdir
from os.path import isfile, join
from shutil import copyfile

def main():
    print("start to pick and rename.")
    # configs
    root_path = '/home/raphael/Desktop/data/ZMY/hand/semantic_segmentation/Images'
    base_dir = '9_25_voc'
    jpeg_dir = 'JPEGImages'
    annotation_dir = 'SegmentationClassPNG'
    dir_save_images = 'new'
    dir_save_annotations = 'new'

    rename_annotations = True
    order = 195
    # traverse the files
    base_path = join(root_path, base_dir)
    jpeg_path = join(base_path, jpeg_dir)
    save_images_path = join(base_path, dir_save_images)
    annotation_path = join(base_path, annotation_dir)
    save_annotations_path = join(base_path, dir_save_annotations)

    for f in listdir(jpeg_path):
        if isfile(join(jpeg_path, f)):
            name, extend_type = f.split('.')
            if extend_type == 'jpg':
                image_src = join(jpeg_path, name+'.jpg')
                image_target = join(save_images_path, str(order)+'.jpg')
                copyfile(image_src, image_target)

                if rename_annotations:
                    annotation_src = join(annotation_path, name+'.png')
                    annotation_target = join(save_annotations_path, str(order)+'.png')
                    copyfile(annotation_src, annotation_target)

                order += 1
                print(order)


if __name__ == "__main__":
    main()