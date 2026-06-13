from PIL import Image
import math, os
from xml.etree import ElementTree as ET


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def make_data_center_txt(xml_dir):
    with open('./datasets/data_center_val.txt', 'a') as f:
        f.truncate(0)
        path = r'datasets/data_val/images'
        xml_names = os.listdir(xml_dir)
        for xml in xml_names:
            xml_path = os.path.join(xml_dir, xml)
            #print(xml.split('.')[0])
            in_file = open(xml_path)
            tree = ET.parse(in_file)
            root = tree.getroot()
            img_path = "D:\pythonSpace\teach_demo\point_regression\data\image\\"
            image_path = img_path + xml.split('.')[0] + ".png"
            polygon = root.find('outputs/object/item/polygon')
            data = []
            c_data = []
            data_str = ''
            for i in polygon:
                data.append(float(i.text))
                data_str = data_str + ' ' + str(i.text)
            for i in range(0, len(data), 2):
                c_data.append((data[i], data[i + 1]))
            # print(c_data[0][0])
            # print(c_data[0][1])

            center_x, center_y = c_data[0][0], c_data[0][1]
            center_x, center_y = math.floor(center_x), math.floor(center_y)
            data_str = os.path.join(path, image_path.split('\\')[-1]) + ' ' + str(center_x) + ' ' + str(center_y) #+ data_str
            f.write(data_str + '\n')


if __name__ == '__main__':
    make_data_center_txt('datasets/data_val/xml')