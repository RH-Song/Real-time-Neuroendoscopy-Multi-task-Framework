#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:54:19 2019
@author: aries
Modify by Raphael Song
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import logging
import json
import cv2

import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

data_file = "/home/raphael/Desktop/data/surgery/seg_aug_data/labels/1.json"
mask_dest_file = "/home/raphael/Desktop/data/surgery/seg_aug_data/masks"
IMG_DIM = 1080

CLASS_COLORS = {"tool": (130, 232, 232)}


def setLogger():
    logger = logging.getLogger('CreateMasks')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


def main():
    try:
        logger = setLogger()

        with open(data_file, 'r') as f:
            tool_segmentations = json.load(f)
            logger.info("%i elements in json file", len(tool_segmentations))

            image_data = tool_segmentations.values()
            print(image_data)

            for img in image_data:

                # create black image on which to draw regions
                mask = np.zeros((IMG_DIM, IMG_DIM), np.uint8)

                # iteratively add shapes
                regions = img["regions"]
                logger.debug("Adding %i regions", len(regions))

                for region in regions:
                    shape = region["shape_attributes"]["name"]
                    bee_type = region["region_attributes"]["class"]
                    # color = CLASS_COLORS[bee_type]
                    color = 255

                    # drawing functions
                    if shape == "circle":

                        x = int(region["shape_attributes"]["cx"])
                        y = int(region["shape_attributes"]["cy"])
                        radius = int(region["shape_attributes"]["r"])

                        cv2.circle(mask, (x, y), radius, color, -1)
                    elif shape == "ellipse":

                        x = int(region["shape_attributes"]["cx"])
                        y = int(region["shape_attributes"]["cy"])
                        rx = int(region["shape_attributes"]["rx"])
                        ry = int(region["shape_attributes"]["ry"])
                        theta = region["shape_attributes"]["theta"]
                        angle = np.degrees(theta)

                        cv2.ellipse(mask, (x, y), (rx, ry), angle, 0, 360, color, -1)

                    elif shape == "polygon":

                        vertices = zip(region["shape_attributes"]["all_points_x"],
                                       region["shape_attributes"]["all_points_y"])

                        # change from list of tuples to list of array
                        vertices = np.array(list(map(list, vertices)), np.int32)

                        cv2.fillPoly(mask, [vertices], color)
                    else:
                        print(shape)
                        raise TypeError(shape, "is not supported")

                plt.imshow(mask)
                plt.show()
                filename = '1.jpg'
                cv2.imwrite(os.path.join(mask_dest_file, filename), mask)


    finally:
        logger.handlers = []

if __name__ == '__main__':
    main()
