'''
2D/3D real-time tracking of surgical instruments based on endoscopie image
'''

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from skimage.filters import frangi
import skimage.draw as draw
import skimage.transform as transform
import matplotlib.pyplot as plt


def get_mask_from_polygon_skimage(image_shape, polygon):
    """Get a mask image of pixels inside the polygon.

    Args:
      image_shape: tuple of size 2.
      polygon: Numpy array of dimension 2 (2xN).
    """
    vertex_row_coords = polygon[:, 1]
    vertex_col_coords = polygon[:, 0]
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, image_shape)
    mask = np.zeros(image_shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def load_images(images_path):
    onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(images_path,onlyfiles[n]))
    return images

def main():
    print("Welcome to the world of endoscope!")
    # load images from fold
    images_path = '/home/raphael/data/surgery/mini_dataset/test-images'
    images = load_images(images_path)

    #cv2.namedWindow('image', 0)
    #cv2.resizeWindow('image', 500, 500)

    for image in images:
        # visualize color images
        # cv2.imshow('origin', image)

        # extract the central part
        # image = image[100:1000,600:1400]
        cv2.imshow('origin', image)

        # convert RGB to CIEBab color space
        CIELab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        cv2.imshow('cielab', CIELab_image)

        # compose to gray image
        l_channel, a_channel, b_channel = cv2.split(CIELab_image)
        cv2.imshow('l', l_channel)
        cv2.imshow('a', a_channel)
        cv2.imshow('b', b_channel)

        # TODO: change chromaticity
        #gray_image = np.sqrt(np.power(b_channel, 2) + np.power(a_channel, 2)).astype(np.uint8)
        gray_image = np.add(np.multiply(a_channel, 0.9), np.multiply(b_channel, 0.1)).astype(np.uint8)

        # disconnect regions(automatic otsu's thresholding)
        # gray_image.convertto(gray_image, cv2.cv_8uc1)
        otsu_th, otsued_image = cv2.threshold(gray_image, 0, 255,
                                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        otsued_image = cv2.bitwise_not(otsued_image)
        cv2.imshow('ostu', otsued_image)

        # distance transform
        # Perform the distance transform algorithm
        dist = cv2.distanceTransform(otsued_image, cv2.DIST_L2, 3)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow('Distance Transform Image', dist)

        # erosion
        structure_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded_dist = cv2.erode(dist, structure_element)
        cv2.imshow('erosion distance transform', eroded_dist)

        # binarized image
        binary_skel = np.rint(eroded_dist * 255)
        binary_skel = np.asarray(binary_skel, dtype=np.uint8)
        cv2.imshow('gray skel', binary_skel)
        ret, binary_skel = cv2.threshold(binary_skel, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow('binary skel', binary_skel)

        # contours detection
        contours, hierarchy = cv2.findContours(binary_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_image = image.copy()
        # cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow('contours image', contours_image)

        # boundingboxes
        mask_shape = gray_image.shape

        # swith image from opencv to skimage
        #contours_image = cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB)
        gray_image.astype(np.float32)
        gray_image = np.divide(gray_image, 255)

        for cnt in contours:
            min_rect = cv2.minAreaRect(cnt)
            w = min_rect[1][0]
            h = min_rect[1][1]
            if w != 0 and h != 0 and max(w, h) / min(w, h) > 1.7:
                min_rect = np.int0(cv2.boxPoints(min_rect))
                # create a mask according to the min rectangle
                cnt_mask = get_mask_from_polygon_skimage(mask_shape, min_rect)
                # Frangi filter(get instruments edges)
                ROI_gray_image = gray_image.copy()
                ROI_gray_image[~cnt_mask] = 1
                # plt.imshow(ROI_gray_image)
                #TODO: change parameters of frangi filter and lines
                frangied_image = frangi(ROI_gray_image)
                # plt.show()
                # Hough line
                lines = transform.probabilistic_hough_line(frangied_image, threshold=10, line_gap=10)
                print(len(lines))
                if len(lines) != 0:
                    cv2.drawContours(contours_image, [min_rect], 0, (255, 0, 0), 2)
                    
                """
                for line in lines:
                    p0, p1 = line
                    cv2.line(contours_image, (p0[0], p1[0]), (p0[1], p1[1]), (0, 255, 0), thickness=3)
                """
        cv2.imshow('bb_image', contours_image)
        print('======')


        # quit the program
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
