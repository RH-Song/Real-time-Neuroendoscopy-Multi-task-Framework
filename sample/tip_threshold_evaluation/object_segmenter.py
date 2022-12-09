"""
Get contours of objects with color infomation
"""

import cv2
import numpy as np

def get_object_segmentation(image, color_space):
    binary_image = []
    if color_space == 'YCrCb':
        # convert to YCrCb color space
        y_image, cr_image, cb_image = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
        binary_image = cr_image
    elif color_space == "gray":
        # convert RGB to gray image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = gray_image
    otsu_th, otsued_image = cv2.threshold(binary_image, 0, 255,
                                          cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsued_image

def get_hand_contours(object_marker):
    """
    Get object contours from image
    :param image: input image
    :return: object contours
    """

    # contours detection
    contours, hierarchy = cv2.findContours(object_marker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    object_contour = contours[0]
    max_area = cv2.contourArea(object_contour)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            object_contour = contour

    return object_contour

def get_skeleton(object_marker):
    # erosion
    structure_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(object_marker.shape, np.uint8)
    erode = np.zeros(object_marker.shape, np.uint8)
    temp = np.zeros(object_marker.shape, np.uint8)

    im = object_marker
    while True:
        erode = cv2.erode(im, structure_element)
        temp = cv2.dilate(erode, structure_element)

        temp = cv2.subtract(im, temp)
        skel = cv2.bitwise_or(skel, temp)
        im = erode.copy()

        if cv2.countNonZero(im) == 0:
            break
    return skel

def get_edge(object_marker):
    object_edge = cv2.Canny(object_marker, 200, 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    object_edge = cv2.dilate(object_edge, kernel)
    return object_edge
