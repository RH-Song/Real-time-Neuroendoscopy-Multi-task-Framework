"""
It is a program used to merge images to video.
"""
import cv2
import numpy as np
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and f.split('.')[1] == 'jpg']

    # for sorting the file names properly
    files.sort(key=lambda x: int(x[:-4]))


    for i in range(len(files)):
        filename = join(pathIn,files[i])
        print(filename)
        # reading each files
        img = cv2.imread(filename)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    h, w, c = frame_array[0].shape
    size = (h, w)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    pathIn = '/home/raphael/Desktop/data/surgery/single-label-seg/hard_cases/specular0'
    pathOut = '/home/raphael/Desktop/data/surgery/single-label-seg/hard_cases/specular0.avi'
    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps)


if __name__ == "__main__":
    main()