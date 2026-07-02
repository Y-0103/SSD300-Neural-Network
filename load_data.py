import torch, random
from pathlib import Path
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

from parse_data import save_data


class Voc_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, image_size = 300, cache_path = 'train_annotations.pt') -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.images_file = self.file_path / 'JPEGImages'
        self.sign_file = self.file_path / 'Annotations'
        self.image_size = image_size

        cache_file = self.file_path / cache_path
        if not cache_file.exists():
            save_data(self.file_path, cache_path)
        self.cache_data = torch.load(
            cache_file, weights_only = False, map_location = 'cpu'
        )
        self.color_jitter = ColorJitter(
            brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.05
        )

    def __len__(self):
        return len(self.cache_data['names'])

    def __getitem__(self, idx):
        image_path = self.images_file / (self.cache_data['names'][idx] + '.jpg')
        image = read_image(str(image_path), ImageReadMode.RGB).float() / 255
        image = F.resize(image, [self.image_size, self.image_size], antialias = True)
        classes_anchors = self.cache_data['classes_anchors'][idx].clone()
        # 水平翻转
        if random.random() < 0.5:
            image = F.hflip(image)
            classes_anchors[:, 1], classes_anchors[:, 3] = \
                1 - classes_anchors[:, 3], 1 - classes_anchors[:, 1]
        # 高斯模糊
        if random.random() < 0.3:
            k = random.choice([3, 5])
            image = F.gaussian_blur(image, kernel_size = k)
        # 转化灰色
        if random.random() < 0.1:
            image = F.rgb_to_grayscale(image, num_output_channels = 3)
        # 转变颜色
        image = self.color_jitter(image)
        return image, classes_anchors


def collate_voc(batch):
    images = torch.stack([x[0] for x in batch], dim = 0)
    anchors_list = [x[1] for x in batch]
    return images, anchors_list
