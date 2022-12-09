"""
It is used to locate tip with laparoscopic images and segmentation results
"""
import data_loader
import object_segmenter
import cv2
import tip_locator
from os.path import join, exists
from os import listdir, mkdir
from sklearn.externals import joblib
from sklearn.svm import SVC
import get_characters_for_svm as SVM_extractor

def show_image(image, TEST_ONE):
    cv2.namedWindow('Object', 0)
    cv2.resizeWindow('Object', 1000, 1000)
    cv2.imshow('Object', image)

    if TEST_ONE:
        cv2.waitKey(0)
        return False
    else:
        if(cv2.waitKey(33) >= 0):
            return False
        else:
            return True


def tip_localization(dir_path, origin_image_dir):
    print('start tip localization')
    result_path = dir_path + '/results_svm'

    dir_path = dir_path + '/mask'
    TEST_ONE = False

    VISIBLE = True
    MASK_OPTIMIZATION = True

    images_filename = data_loader.load_images_from_dir(dir_path, 'png')

    if not exists(result_path):
        mkdir(result_path)
    results_name = result_path + '/results.txt'
    results_txt = open(results_name, 'w')

    for image_filename in images_filename:
        #print('filename')
        #print(image_filename)

        results_txt.write('filename\n')
        results_txt.write(image_filename+'\n')

        mask_image = cv2.imread(join(dir_path, image_filename))
        name, extend = image_filename.split('.')
        BGR_image = cv2.imread(join(origin_image_dir, name+'.jpg'))

        #image = cv2.resize(origin_image, (640, 480))

        object_marker = object_segmenter.get_object_segmentation(mask_image, 'gray')

        contours, hierarchy = cv2.findContours(object_marker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        visi_image = BGR_image

        svclassifier = joblib.load("trained_models/gs.m")
        cv2.drawContours(visi_image, contours, -1, (255, 0, 0), 10)
        for contour in contours:
            contour_label = 0
            if MASK_OPTIMIZATION:
                # classify by svm
                normalized_entry_point = SVM_extractor.get_normalized_entry_point_dis(BGR_image, contour)
                normalized_area = SVM_extractor.get_normalized_area(BGR_image, contour)
                contour_label = svclassifier.predict([[normalized_entry_point, normalized_area]])
            else:
                contour_label = 1
            if contour_label == 1:
                tip_point = tip_locator.locate_tip(BGR_image, contour)
                # print tip
                #print('tip')
                #print(tip_point)

                results_txt.write('tip\n')
                results_txt.write(str(tip_point)+'\n')

                if VISIBLE:
                    cv2.circle(visi_image, (tip_point[0], tip_point[1]), 10, (0, 255, 255), -1)

        if VISIBLE:
            cv2.imwrite(join(result_path, image_filename), visi_image)
        #show_image(object_marker, TEST_ONE)
        #show_image(visi_image, TEST_ONE)

if __name__ == '__main__':

    # configs
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/deeplabv3"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/fcn8s"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/ground_truth"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp_svm"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/visible/deeplabv3"
    origin_path = "/home/raphael/Desktop/test/test/fcn8s_00-59-35-new"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/visible/gt"
    #origin_image_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_dataset/img"
    origin_image_dir = "/home/raphael/Desktop/data/surgery/ysm/B-demos/00-59-35-new"

    tip_localization(origin_path, origin_image_dir)
