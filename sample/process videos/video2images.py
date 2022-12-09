import cv2
import numpy as np

def main():
    print("Welcome to use the script to split video into images.")

    path_prefix = '/home/raphael/Desktop/data/ZMY/data/'

    cap = cv2.VideoCapture(path_prefix + '2020-11-26-151946.mp4')

    image_counter = 0
    num_of_image = 1
    # if cap.isOpened():
    rval, frame = cap.read()

    # get FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('FPS:' + str(fps))

    # start_point = (0*3600+59*60+35) * fps
    # end_point = (0*3600+59*60+50) * fps
    #
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_point-2)
    # print('set start point down.')

    while rval:
        rval, frame = cap.read()
        # choose and save one image

        if num_of_image >= 0:
            #cv2.imshow('frame', frame)
            cv2.imwrite(path_prefix + 'chessboard/'+str(num_of_image)+'.png', frame)
            #cv2.waitKey(0)
            print(num_of_image)

        num_of_image += 1
        # if num_of_image == end_point - start_point:
        #     break
        # if cv2.waitKey(25) & 0xFF == ord('q'):
            # break

    cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()