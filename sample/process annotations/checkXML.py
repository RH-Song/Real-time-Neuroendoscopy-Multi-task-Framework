# -*- coding: utf-8 -*-
# 可视化 标注结果，并按照 标注的 人脸框 和 人体框 个数 存放图片，便于检查
import os
import cv2
import shutil
import xml.etree.ElementTree as ET

def checkxml(xml_f):
    in_file = open(xml_f)
    tree=ET.parse(in_file)
    in_file.close()
    root = tree.getroot()

    num = len(root.findall('object'))
    count = 0
    for obj in root.iter('object'):
        cls = obj.find('difficult').text
        if cls == "1":
            count = count + 1
    if num == count:
        print(xml_f)

xmldir = "/home/raphael/Desktop/data/surgery/surgery_VOCstyle/Annotations"
for xmlname in os.listdir(xmldir):
    xml_f = os.path.join(xmldir, xmlname)
    checkxml(xml_f)