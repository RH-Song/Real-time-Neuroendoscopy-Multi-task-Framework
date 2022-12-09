"""
Use Key broad to take photos
author: Raphael Song
"""
import cv2
from os.path import join, exists
from os import mkdir

def show_video(image):
    cv2.namedWindow('video', 0)
    cv2.resizeWindow('video', 500, 500)

    cv2.imshow('video', image)

def show_photo(image):
    cv2.namedWindow('photo', 0)
    cv2.resizeWindow('photo', 500, 500)

    cv2.imshow('photo', image)

if __name__ == '__main__':
    print("Weclome to camera loader")

    # configs:square 23mm
    root_dir = '/home/raphael/Desktop/data/surgery/camera_calibration/imgs'
    dir_path = join(root_dir, "imgs")
    if not exists(dir_path):
        mkdir(dir_path)

    camera_idx = 0
    photo_index = 0
    frame_counter = 0
    # load data
    USE_CAMERA = True
    KEYBOARD_MODE = True

    if USE_CAMERA:
        cap = cv2.VideoCapture(camera_idx)

        # test camera is opened or not
        rval = False
        if cap.isOpened():
            rval, frame = cap.read()
        else:
            print("Camera is not opened.")

        while rval:
            rval, frame = cap.read()
            frame_counter += 1
            key = cv2.waitKey(33)
            if KEYBOARD_MODE:
                # space
                if key == 32:
                    show_photo(frame)
                    cv2.imwrite(join(dir_path, str(photo_index)+".jpg"), frame)
                    print(photo_index)
                    photo_index += 1
            else:
                if frame_counter >= 0:
                    frame_counter = 0
                    show_photo(frame)
                    cv2.imwrite(join(dir_path, str(photo_index)+".jpg"), frame)
                    print(photo_index)
                    photo_index += 1

            # tab: switch KEYBROAD value
            if key == 9:
                print('key == 9')
                print(KEYBOARD_MODE)
                if KEYBOARD_MODE:
                    KEYBOARD_MODE = False
                else:
                    KEYBOARD_MODE = True
            # ESC
            if key == 27:
                break
            show_video(frame)

        cap.release()
        cv2.destroyAllWindows()