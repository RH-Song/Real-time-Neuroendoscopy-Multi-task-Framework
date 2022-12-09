"""
It is used to evaluate the average width of tools.
num of tool: 10841, average width: 178.71952270696156
"""
from os.path import join
import cv2
import numpy
import object_segmenter
from os import listdir
import numpy as np

if __name__ == '__main__':
    print('start evaluate average width of tools.')
    # load ground true
    segmentation_root_path = "/home/raphael/Desktop/data/surgery/surgery_seg_dataset"
    origin_image_dir = join(segmentation_root_path, 'img')
    seg_gt_dir = join(segmentation_root_path, 'ground_true')

    gt_objs = listdir(seg_gt_dir)

    tool_num = 0
    total_width = 0
    #centers = []
    rotations = []
    widths = []
    num_of_tool = 0
    AF_num = 0
    BDEH_num = 0
    C_num = 0
    h_num = 0
    w_num = 0
    one_num = 0
    two_num = 0
    three_num = 0
    four_num = 0
    for gt_obj in gt_objs:
        print(gt_obj)
        name, _ = gt_obj.split('.')
        num, _ = name.split('_')
        mask_image = cv2.imread(join(seg_gt_dir, gt_obj))
        # found segs
        gt_gray = object_segmenter.get_object_segmentation(mask_image, 'gray')
        contours, hierarchy = cv2.findContours(gt_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # fit segs with minrect or ellips

        index_of_file = int(num)
        len_countours = len(contours)
        if len_countours == 1:
            one_num += 1
        elif len_countours == 2:
            two_num += 1
        elif len_countours >= 3:
            three_num += 1
        for contour in contours:
            tool_num += 1

            if index_of_file >= 861 and index_of_file <= 1214:
                AF_num += 1
            elif index_of_file >= 2631 and index_of_file <= 3124:
                BDEH_num += 1
            elif index_of_file >= 802 and index_of_file <= 860:
                C_num += 1
            elif index_of_file >= 2103 and index_of_file <= 2630:
                BDEH_num += 1
            elif index_of_file >= 1215 and index_of_file <= 1694:
                BDEH_num += 1
            elif index_of_file >= 1695 and index_of_file <= 2102:
                AF_num += 1
            elif index_of_file >= 1 and index_of_file <= 801:
                BDEH_num += 1

            rect = cv2.minAreaRect(contour)
            rotation = 0
            if rect[0][0] < 540 and rect[0][1] < 540:
                rotation = -270 + int(rect[2])
            elif rect[0][0] > 540 and rect[0][1] < 540:
                rotation = -90 + int(rect[2])
            elif rect[0][0] > 540 and rect[0][1] > 540:
                rotation = 0 + int(rect[2])
            elif rect[0][0] < 540 and rect[0][1] > 540:
                rotation = -180 + int(rect[2])
            rotations.append(-1 * rotation)
            w = rect[1][0]
            h = rect[1][1]
            widths.append(int(w))
            if h > w:
                h_num += 1
            else:
                w_num += 1
            total_width += w
            num_of_tool += 1
            #print(rect)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(mask_image, [box], 0, (255, 0, 0), 5)

    print(AF_num)
    print(BDEH_num)
    print(C_num)
    r_hist, r_edges = np.histogram(rotations)
    print(r_hist)
    print(r_edges)
    w_hist, w_edges = np.histogram(widths)
    print(w_hist)
    print(w_edges)
    print(num_of_tool)
    print(h_num)
    print(w_num)
    print(one_num)
    print(two_num)
    print(three_num)
    print(four_num)
    average_width = total_width / tool_num
    print('num of tool: {}, average width: {}'.format(tool_num, average_width))
        # Image visulization
        # cv2.namedWindow('Object', 0)
        # cv2.resizeWindow('Object', 1000, 1000)
        # cv2.imshow('Object', mask_image)
        # cv2.waitKey(0)
        # break
