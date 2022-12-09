import math

def find_farthest_point(center_point, contour):
    farthest_dis = 0
    farthest_point = center_point
    for point in contour:
        point = point[0]
        euclidean_dis = math.sqrt(math.pow(point[0]-center_point[0], 2) +
                                  math.pow(point[1]-center_point[1], 2))
        if euclidean_dis > farthest_dis:
            farthest_point = point
            farthest_dis = euclidean_dis
    return farthest_point, euclidean_dis

def find_entry_point(image, contour):
    w,h,d = image.shape
    center_point = [int(h/2), int(w/2)]
    farthest_point, euclidean_dis = find_farthest_point(center_point, contour)
    return farthest_point, euclidean_dis

def tip_rough_position(image, contour):
    # Entry Point: find the point farthest away from tip
    entry_point, euclidean_dis = find_entry_point(image, contour)

    # find the tip point
    center_point = entry_point
    farthest_point, euclidean_dis = find_farthest_point(center_point, contour)
    return farthest_point

def locate_tip(image, contour):
    rough_position = tip_rough_position(image, contour)
    #print(farthest_point)
    farthest_point = rough_position
    return farthest_point
