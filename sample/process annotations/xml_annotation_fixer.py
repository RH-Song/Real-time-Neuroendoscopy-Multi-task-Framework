"""
It is a program used to change the wrong filename
in xml files to the right name which is the same name as the xml name
"""
from os import listdir
from os.path import isfile, join
from xml.dom import minidom

def main():
    print('Start changing the filename in xml files...')

    # configs
    dir_path = '/data/home/usi/data/surgery/surgery_VOCstyle/Annotations'

    # taverse the dir
    files = listdir(dir_path)
    for file_obj in files:
        if isfile(join(dir_path, file_obj)):
            name, extended_name = file_obj.split('.')
            if extended_name == 'xml':

                # parse the xml file
                xml = minidom.parse(join(dir_path, file_obj))
                root = xml.documentElement
                filename = root.getElementsByTagName('filename')
                filename = filename[0]
                filename.firstChild.data = name + '.jpg'

                with open(join(dir_path, file_obj), 'w') as fh:
                    xml.writexml(fh)
                    print(name + 'write successful.')

if __name__ == "__main__":
    main()