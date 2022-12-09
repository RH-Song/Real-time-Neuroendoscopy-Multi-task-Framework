import numpy as np
import os
import cv2
import json

class labelme_convertor():
    def __init__(self, image_name):
        self.save_json_fold = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/json/"
        self.img_fold = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/masks/"
        self.name = image_name[:-4]

        self.imagePath = os.path.join(self.img_fold, image_name)
        self.img = cv2.imread(self.imagePath)

        self.save_json_path = os.path.join(self.save_json_fold, self.name+'.json')

        self.version = "3.16.7"
        self.flags = {}
        self.shapes = []
        self.lineColor = [0,255,0,128]
        self.fillColor = [255,0,0,128]
        self.imageWidth, self.imageHeight, self.imageChannel = self.img.shape

        self.label = "tool"
        self.points = []
        self.shape_type = "polygon"
        self.shapeLineColor = None
        self.shapefillColor = None

        self.get_instance_contours()
        self.get_labelme_json()

        # self.cmap = self.label_colormap()

    def get_instance_contours(self):
        # color_map
        color_map = {}
        # segmentation: parts
        seg_parts = []
        for c in range(self.imageWidth):
            for r in range(self.imageHeight):
                color = (self.img[c][r][0], self.img[c][r][1], self.img[c][r][2])
                if color != (0,0,0):
                    index = color_map.get(color, 'never')
                    if index == 'never':
                        new_index = len(color_map)
                        color_map[color] = new_index
                        new_shape = []
                        new_shape.append([c,r])
                        seg_parts.append(new_shape)
                    else:
                        seg_parts[index].append([c,r])

        # find contours
        # append into points
        for seg in seg_parts:
            instance_mask = np.zeros((self.imageWidth, self.imageHeight), dtype="uint8")
            for point in seg:
                instance_mask[point[0]][point[1]] = 255
            # instance_binary = cv2.threshold(instance_mask, 100, 255, type=0)
            cnts, _ = cv2.findContours(instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            points = []
            for cnt in cnts:
                for p in cnt:
                    points.append(p[0].tolist())
            self.shapes.append(points)

    def get_labelme_json(self):
        labelme_data = {}
        labelme_data["version"] = self.version
        labelme_data["flags"] = self.flags
        labelme_shapes = []
        for shape in self.shapes:
            shape_obj = {}
            shape_obj["label"] = self.label
            shape_obj["line_color"] = self.shapeLineColor
            shape_obj["fill_color"] = self.shapefillColor
            # points = []
            # for p in shape:
            #     points.append([p[0], p[1]])
            shape_obj["points"] = shape
            shape_obj["shape_type"] = self.shape_type
            shape_obj["flags"] = self.flags
            labelme_shapes.append(shape_obj)
        labelme_data["shapes"] = labelme_shapes
        labelme_data["lineColor"] = self.lineColor
        labelme_data["fillColor"] = self.fillColor
        labelme_data["imagePath"] = self.imagePath
        labelme_data["imageData"] = None
        labelme_data["imageHeight"] = self.imageHeight
        labelme_data["imageWidth"] = self.imageWidth

        json.dump(labelme_data, open(self.save_json_path, 'w'), indent=4)

    def label_colormap(self, n_label=256, value=None):
        """Label colormap.

        Parameters
        ----------
        n_labels: int
            Number of labels (default: 256).
        value: float or int
            Value scale or value of label color in HSV space.

        Returns
        -------
        cmap: numpy.ndarray, (N, 3), numpy.uint8
            Label id to colormap.

        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(0, n_label):
            id = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
                g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
                b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        return cmap

if __name__ == '__main__':
    img_fold = "/home/raphael/Desktop/data/surgery/single-label-seg/allmask/augmented/voc_style_2/masks/"
    count = 1
    for obj in os.listdir(img_fold):
        print(obj)
        labelme_convertor(obj)
        print(count)
        count += 1
