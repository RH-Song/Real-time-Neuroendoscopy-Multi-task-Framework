"""
It is a program that used to parse xml annotations to tracking txt
"""
from os import listdir
from os.path import join, isfile
from xml.dom import minidom

def main():
    print("start parsing xml to txt...")

    # configs
    annotation_dir = "/home/raphael/Desktop/data/surgery/a/tracking/02-02-26/Annotations"
    txt_path = "/home/raphael/Desktop/data/surgery/a/tracking/02-02-26/02-02-26.txt"

    txt = open(txt_path, 'a')

    xml_list = listdir(annotation_dir)
    xml_list.sort(key= lambda x:int(x[:-4]))
    for xml_obj in xml_list:
        xml_file = join(annotation_dir, xml_obj)
        if isfile(xml_file):
            name, extend = xml_obj.split('.')
            # parse xml. extract the filename as frame name, object name and calculate center point, width and hight
            if extend == "xml":
                root = minidom.parse(xml_file)
                frame_names = root.getElementsByTagName("filename")
                frame_name, extend = frame_names[0].firstChild.data.split('.')
                print(frame_name)
                names = root.getElementsByTagName("name")
                xmins = root.getElementsByTagName("xmin")
                xmaxs = root.getElementsByTagName("xmax")
                ymins = root.getElementsByTagName("ymin")
                ymaxs = root.getElementsByTagName("ymax")

                for i in range(0, len(names)):
                    name = int(names[i].firstChild.data)
                    xmin = int(xmins[i].firstChild.data)
                    xmax = int(xmaxs[i].firstChild.data)
                    ymin = int(ymins[i].firstChild.data)
                    ymax = int(ymaxs[i].firstChild.data)
                    print("{}, {}, {}, {}, {}".format(name, xmin, xmax, ymin, ymax))
                    center_point_x = int((xmin + xmax) / 2)
                    center_point_y = int((ymin + ymax) / 2)
                    width = xmax - xmin
                    hight = ymax - ymin
                    print("{}, {}, {}, {}".format(center_point_x, center_point_y, hight, width))
                    txt.write("{} {} {} {} {} {}\n".format(frame_name, name, center_point_x, center_point_y, width, hight))

            else:
                print("Not xml file.")
        else:
            print("Not a file.")
    txt.close()


if __name__ == '__main__':
    main()