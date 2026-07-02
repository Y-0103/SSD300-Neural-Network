import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
from config import CLASSES


def parse_xml(xml_path: str) -> Tensor:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    classes_anchors = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASSES:
            continue
        anchor = obj.find('bndbox')
        xmin = float(anchor.find('xmin').text)
        ymin = float(anchor.find('ymin').text)
        xmax = float(anchor.find('xmax').text)
        ymax = float(anchor.find('ymax').text)
        names = CLASSES.index(name) + 1
        classes_anchors.append([names, xmin / width, ymin / height,
                        xmax / width, ymax / height])

    device = torch.device('cpu')
    classes_anchors = torch.tensor(classes_anchors, device = device)
    return classes_anchors, (width, height)


def save_data(file_path, cache_path):
    file_path = Path(file_path)
    sign_file = file_path / 'Annotations'
    images_paths = sorted((file_path / 'JPEGImages').glob('*.jpg'))
    sizes = []
    names = []
    classes_anchors = []
    for image_path in tqdm(images_paths, desc = '解析 XML 文件'):
        sign_path = sign_file / (image_path.stem + '.xml')
        classes_anchor, (width, height) = parse_xml(str(sign_path))
        classes_anchors.append(classes_anchor)
        sizes.append((width, height))
        names.append(image_path.stem)
    torch.save({
        'names': names,
        'classes_anchors': classes_anchors,
        'sizes': sizes,
    }, file_path / cache_path)
