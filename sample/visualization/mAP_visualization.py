"""
It is a program used to draw loss from log file
"""
# legend
import numpy as np
import matplotlib.pyplot as plt

def main():
    print('start to visualize the mAP...')
    # configs
    result1_path= '/home/raphael/Downloads/results_23.txt'
    result2_path= '/home/raphael/Downloads/results_21.txt'
    result3_path= '/home/raphael/Downloads/results_21_4.txt'
    result4_path= '/home/raphael/Downloads/result_0904.txt'

    # open file
    log1 = open(result1_path, 'r')
    log2 = open(result2_path, 'r')
    log3 = open(result3_path, 'r')
    log4 = open(result4_path, 'r')

    # read line
    log1_lines = log1.readlines()
    log2_lines = log2.readlines()
    log3_lines = log3.readlines()
    log4_lines = log4.readlines()

    # data
    mAP23 = []
    mAP21 = []
    mAP21_4 = []
    mAP0904 = []

    common_range = 6
    iter_range = 1
    for log_line in log1_lines:
        words = log_line.split(' ')
        for i in range(0, len(words)):
            if words[i] == 'mAP:' and iter_range < common_range:
                mAP23.append(float(words[i+1]) * 100)
                iter_range += 1

    iter = [model for model in range(2000, 2000*common_range, 2000)]
    mAP23_line = plt.plot(iter, mAP23, 'r-o', label='Transfer Learning', linewidth = "1")
    plt.grid(True, linestyle = "-", color = "gray", linewidth = "1")

    iter_range = 1
    for log_line in log2_lines:
        words = log_line.split(' ')
        for i in range(0, len(words)):
            if words[i] == 'mAP:' and iter_range < common_range:
                mAP21.append(float(words[i+1]) * 100)
                iter_range += 1

    iter21 = [model for model in range(2000, 2000*common_range, 2000)]
    mAP21_line = plt.plot(iter21, mAP21, 'b-s', label='origin', linewidth = "1")

    iter_range = 1
    for log_line in log3_lines:
        words = log_line.split(' ')
        for i in range(0, len(words)):
            if words[i] == 'mAP:' and iter_range < common_range:
                mAP21_4.append(float(words[i+1]) * 100)
                iter_range += 1

    iter21_4 = [model for model in range(2000, 2000*common_range, 2000)]
    mAP21_4_line = plt.plot(iter21_4, mAP21_4, 'g-^', label='Data Augmentation', linewidth = "1")

    iter_range = 1
    for log_line in log4_lines:
        words = log_line.split(' ')
        for i in range(0, len(words)):
            if words[i] == 'mAP:' and iter_range < common_range:
                mAP0904.append(float(words[i+1]) * 100)
                iter_range += 1

    iter0904 = [model for model in range(2000, 2000*common_range, 2000)]
    mAP0904_line = plt.plot(iter0904, mAP0904, '-*', color='black', label='pre-train weight', linewidth = "1")

    plt.xlabel('iter')
    plt.ylabel('mAP')
    plt.legend()
    plt.show()

    print('done.')

if __name__ == '__main__':
    main()