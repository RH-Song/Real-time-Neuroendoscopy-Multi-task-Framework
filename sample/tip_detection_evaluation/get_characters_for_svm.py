"""
Prepare data to train SVM to classify segmentation
"""
import pandas as pd
import numpy as np
import data_loader
import object_segmenter
import cv2
from os.path import join
import tip_locator
import evaluator

def get_normalized_entry_point_dis(image, contour):
    img_h, img_w, d = image.shape
    entry_point, euclidean_dis = tip_locator.find_entry_point(image, contour)
    normalized_dis = euclidean_dis / (img_h / 2)
    return normalized_dis

def get_normalized_area(image, contour):
    img_h, img_w, d = image.shape
    total_area = img_h * img_w
    contour_area = cv2.contourArea(contour)
    normalized_area = contour_area / total_area
    return normalized_area

if __name__ == '__main__':
    print('start get characters.')

    origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/"
    root_paths = ['psp', 'deeplabv3', 'fcn8s', 'maskrcnn']

    segmentation_root_path = "/home/raphael/Desktop/data/surgery/surgery_seg_dataset"
    origin_image_dir = join(segmentation_root_path, 'img')
    seg_gt_dir = join(segmentation_root_path, 'ground_true')
    IOU_THRESHOLD = 0.50

    image_names = []
    entry_point_distances = []
    mask_areas = []
    labels = []
    num_of_true = 0
    total_num = 0

    for root in root_paths:
        root_path = join(origin_path, root)
        mask_path = root_path + "/mask"

        images_filename = data_loader.load_images_from_dir(mask_path, 'png')

        for image_filename in images_filename:
            print('filename')
            print(image_filename)

            # Get name
            name, extended_type = image_filename.split('.')
            image_names.append(name)

            # Get the origin image
            name, extend = image_filename.split('.')
            BGR_image = cv2.imread(join(origin_image_dir, name+'.jpg'))

            # Get ground true
            gt_filename = join(seg_gt_dir, name+'_json.png')
            mask_image = cv2.imread(gt_filename)
            # found segs
            gt_gray = object_segmenter.get_object_segmentation(mask_image, 'gray')
            contours, hierarchy = cv2.findContours(gt_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Get boxes of ground truth
            gt_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Format: left top right bottom
                box_rect = np.int0([y-w/2, x-h/2, y+w/2, x+h/2])
                gt_boxes.append(box_rect)

            # Get the prediction
            mask_image = cv2.imread(join(mask_path, image_filename))
            # Transform to gray image
            object_marker = object_segmenter.get_object_segmentation(mask_image, 'gray')
            # Get the contours of preditions
            contours, hierarchy = cv2.findContours(object_marker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            visi_image = BGR_image
            img_h, img_w, d = visi_image.shape
            total_area = img_h * img_w

            # Characters: distance from entry point to center point, hull area, ratio of length and width.
            for contour in contours:
                # Get entry point
                normalized_dis = get_normalized_entry_point_dis(visi_image, contour)
                entry_point_distances.append(normalized_dis)

                # Get contour area
                normalized_area = get_normalized_area(visi_image, contour)
                mask_areas.append(normalized_area)

                # Get Label according to iou
                x, y, w, h = cv2.boundingRect(contour)
                # Format: left top right bottom
                box_rect = np.int0([y-w/2, x-h/2, y+w/2, x+h/2])
                #print(box_rect)
                highest_iou = evaluator.get_highest_iou(box_rect, gt_boxes)
                #print(highest_iou)
                label = 0
                if highest_iou > IOU_THRESHOLD:
                    label = 1
                    num_of_true += 1
                #print(label)
                total_num += 1
                labels.append(label)

    data_frame = pd.DataFrame({'entry_point_distance':entry_point_distances, 'area':mask_areas, 'label':labels})

    data_frame.to_csv("tip_data.csv",index=False,sep=',')
    print(num_of_true)
    print(total_num)