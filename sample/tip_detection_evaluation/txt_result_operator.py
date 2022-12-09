"""
It used to format raw results to results easy to be read by program
"""
from os import path, mkdir, listdir
from os.path import join, exists
import cv2
import numpy as np

def format_raw_results(file_dir):
    print('start format_raw_results.')
    file_dir = file_dir + '/results'
    file_name = 'results.txt'
    save_name = 'results_refact.txt'

    raw_results = open(join(file_dir, file_name), 'r')
    organized_results = open(join(file_dir, save_name), 'w')
    result_lines = raw_results.readlines()

    for line in result_lines:
        #print(line)
        new_line = ''

        line_len = len(line)
        #print('lenght: {}'.format(line_len))
        if line_len == 1:
            continue

        last_word = ''
        for word in line:
            #print(word)
            if word != '[' and word != ']' and word != ',':
                if word == ' ':
                    if last_word != ' ' and last_word != '[':
                        new_line += word
                else:
                    new_line += word
            last_word = word
        #print('new line: {}'.format(new_line))
        organized_results.write(new_line)
    raw_results.close()
    organized_results.close()
    print('finish.')

def split_result(origin_file_dir):
    print('start split.')
    origin_file_dir = origin_file_dir + '/results'
    file_dir = origin_file_dir + '/singles'

    results_file = 'results_refact.txt'
    results = open(join(origin_file_dir, results_file), 'r')

    if not exists(file_dir):
        mkdir(file_dir)

    result_lines = results.readlines()
    results_len = len(result_lines)
    print(results_len)
    # filename:1; BDD:2; contours:3; tip:4
    STATE = 0

    result_name = ''
    i = 0
    while i < results_len:
        line = result_lines[i]
        words = line.split(' ')
        if words[0] == 'filename\n':
            STATE = 1
        elif words[0] == 'BDDs\n':
            STATE = 2
        elif words[0] == 'contour\n':
            STATE = 3
        elif words[0] == 'tip\n':
            STATE = 4

        # filename
        if STATE == 1:
            i += 1
            filename = result_lines[i]
            name = filename.split('.')
            result_name = name[0] + '.txt'
            temp_result = open(join(file_dir, result_name), 'a')
            temp_result.write('filename\n')
            temp_result.write(filename)
            # print(filename)
            temp_result.close()

        # BDD
        elif STATE == 2:
            temp_result = open(join(file_dir, result_name), 'a')
            temp_result.write('BDDs\n')
            for j in range(1, 3):
                i += 1
                point_line = result_lines[i]
                temp_result.write(point_line)
            temp_result.close()

        # Contour
        elif STATE == 3:
            temp_result = open(join(file_dir, result_name), 'a')
            temp_result.write('contour\n')
            while True:
                i += 1
                point_line = result_lines[i]
                if point_line != 'tip\n':
                    temp_result.write(point_line)
                else:
                    i -= 1
                    break
            temp_result.close()

        # Tip
        elif STATE == 4:
            temp_result = open(join(file_dir, result_name), 'a')
            temp_result.write('tip\n')
            i += 1
            point_line = result_lines[i]
            temp_result.write(point_line)
            temp_result.close()

        i += 1

def get_point(line):
    cordinates = line.split(' ')
    cordinates[0] = int(cordinates[0][:])
    cordinates[1] = int(cordinates[1][:-1])
    return [cordinates[0], cordinates[1]]

def get_tip(result):
    lines = result.readlines()
    line_len = len(lines)
    tip_points = []
    TIP = False
    for line in lines:
        if TIP:
            tip_points.append(get_point(line))
            TIP = False
        elif line == 'tip\n':
            TIP = True
    return tip_points

def tip_visulization():
    # config
    image_dir = '/home/raphael/Desktop/data/surgery/tip_detection/results_10_22/Images'
    results_dir = '/home/raphael/Desktop/data/surgery/tip_detection/results_10_22/single_results'

    result_objs = listdir(results_dir)
    for result_obj in result_objs:
        name, type = result_obj.split('.')
        image_obj = name + '.jpg'
        image = cv2.imread(join(image_dir, image_obj))
        result = open(join(results_dir, result_obj), 'r')
        tip_points = get_tip(result)
        result.close()
        for tip_point in tip_points:
            cv2.circle(image, (int(tip_point[0]), int(tip_point[1])), 10, (255, 0, 255), -1)
        cv2.imshow('image', image)
        cv2.waitKey(0)

def find_mask():
    #config
    image_dir = '/home/raphael/Desktop/data/surgery/tip_detection/results_10_23_afternoon/Images'
    mask_dir = '/home/raphael/Desktop/data/surgery/tip_detection/results_10_23_afternoon/Masks'

    image_objs = listdir(image_dir)
    for image_obj in image_objs:
        image = cv2.imread(join(image_dir, image_obj))

        h, w, d = image.shape
        mask = np.zeros((w, h), dtype=np.uint8)

        for col in range(0, w):
            for row in range(0, h):
                pixel = image[row][col]
                #print(pixel)
                if pixel[0] == 255 and pixel[1] == 0 and pixel[2] == 0:
                    #if pixel[0] == 255:
                    print('yes')
                    mask[row][col] = 255

        cv2.imshow('mask', mask)
        cv2.waitKey(0)


if __name__ == '__main__':

    #origin_file_dir = '/home/raphael/Desktop/data/surgery/surgery_seg_results/results_11-1'
    #origin_file_dir = '/home/raphael/Desktop/data/surgery/surgery_seg_results/deeplabv3'
    #origin_file_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/ground_truth"
    #origin_file_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/fcn8s"
    #origin_file_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp"
    origin_file_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp_svm"

    format_raw_results(origin_file_dir)
    split_result(origin_file_dir)
    #tip_visulization()
#    print('tip visulization')
