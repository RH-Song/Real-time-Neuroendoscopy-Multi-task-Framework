"""
It is a program used to draw loss from log file
"""
# legend
import numpy as np
import matplotlib.pyplot as plt

def main():
    print('start to visualize the loss...')
    # configs
    log_path = '/data/home/usi/data/surgery/logs/log.txt'

    # open file
    log = open(log_path, 'r')
    # read line
    log_lines = log.readlines()

    # data
    iter = []
    loss = []
    loss_classifier = []
    loss_box_reg = []
    loss_objectness = []
    loss_rpn_box_reg = []
    lr = []
    iter_range = 0
    for log_line in log_lines:
        words = log_line.split(' ')
        for i in range(0, len(words)):
            if words[i] == 'iter:':
                iter.append(float(words[i+1]))
                iter_range = float(words[i+1])
            elif words[i] == 'loss:':
                loss.append(float(words[i+1]))
                """
                loss_classifier.append(float(words[i+5]))
                loss_box_reg.append(float(words[i+7]))
                loss_objectness.append(float(words[i+9]))
                loss_rpn_box_reg.append(float(words[i+11]))
                """
            elif words[i] == 'lr:':
                lr.append(float(words[i+1]))
        if iter_range > 1000:
            break

    l_loss = plt.plot(iter, loss, 'r--', label='loss')
    l_lr = plt.plot(iter, lr, 'b--', label='lr')
    plt.xlabel = 'iter'
    plt.ylabel = 'loss'
    plt.legend()
    plt.show()

    print('done.')

if __name__ == '__main__':
    main()