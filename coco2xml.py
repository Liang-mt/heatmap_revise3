import os
import json
import xml.etree.ElementTree as ET


def json_to_xml(json_path, save_path):
    # 读取.json文件内容
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 创建根元素
    root = ET.Element('doc')

    # 创建<path>元素并添加内容
    path = ET.SubElement(root, 'path')
    path.text = 'D:\\pythonSpace\\teach_demo\\point_regression\\data\\image\\0.png'

    # 创建<outputs>元素
    outputs = ET.SubElement(root, 'outputs')

    # 创建<object>元素
    obj = ET.SubElement(outputs, 'object')

    # 创建<item>元素
    item = ET.SubElement(obj, 'item')

    # 创建<name>元素并添加内容
    name = ET.SubElement(item, 'name')
    name.text = 'point'

    # 创建<polygon>元素
    polygon = ET.SubElement(item, 'polygon')

    # 创建<x1>元素并添加内容
    x1 = ET.SubElement(polygon, 'x1')
    x1.text = str(data['shapes'][0]['points'][0][0])

    # 创建<y1>元素并添加内容
    y1 = ET.SubElement(polygon, 'y1')
    y1.text = str(data['shapes'][0]['points'][0][1])

    # 创建<time_labeled>元素并添加内容
    time_labeled = ET.SubElement(root, 'time_labeled')
    time_labeled.text = '1636786273734'

    # 创建<labeled>元素并添加内容
    labeled = ET.SubElement(root, 'labeled')
    labeled.text = 'true'

    # 创建<size>元素
    size = ET.SubElement(root, 'size')

    # 创建<width>元素并添加内容
    width = ET.SubElement(size, 'width')
    width.text = str(data['imageWidth'])

    # 创建<height>元素并添加内容
    height = ET.SubElement(size, 'height')
    height.text = str(data['imageHeight'])

    # 创建<depth>元素并添加内容
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'  # 假设深度固定为3

    # 创建ElementTree对象
    tree = ET.ElementTree(root)

    # 获取.json文件名，作为.xml文件名
    json_file_name = os.path.basename(json_path)
    xml_file_name = os.path.splitext(json_file_name)[0] + '.xml'

    # 拼接保存路径和文件名
    save_file_path = os.path.join(save_path, xml_file_name)

    # 将ElementTree对象写入.xml文件
    tree.write(save_file_path)


def convert_folder(folder_path, save_folder_path):
    # 遍历文件夹中的所有.json文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)

            # 将.json文件转换为.xml文件
            json_to_xml(json_path, save_folder_path)


# 定义源文件夹和目标文件夹路径
json_folder_path = r"C:/Users/28645/Desktop/keypoints/val"
xml_save_folder_path = r"C:/Users/28645/Desktop/keypoints/xml"

# 将文件夹中的所有.json文件转换为.xml文件
convert_folder(json_folder_path, xml_save_folder_path)