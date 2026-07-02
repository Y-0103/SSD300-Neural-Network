import torch
from torch.nn import functional as F2
import torchvision.transforms.functional as F
from torchvision.io import read_image, ImageReadMode

import numpy as np
import matplotlib.pyplot as plt

from ssd_net import SSD
from anchors_opt import nms, decode_offeset
from config import CLASSES, CHECKPOINT_PATH, IMAGE_SIZE


class Predict:
    def __init__(self, net: Modules, checkpoint: dict = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net
        self.checkpoint = checkpoint
        self.image_size = IMAGE_SIZE
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net = self.net.to(self.device)
        self.net.eval()

    def run(self, image: Tensor, origin_width: int, origin_height: int,
            probability = 0.01, nms_iou = 0.45) -> tuple:
        self.image = image
        self.origin_height = origin_height
        self.origin_width = origin_width
        X = self.image.unsqueeze(0).to(self.device)
        init_anchors, pred_classes, pred_offeset = self.net(X)
        pred_classes = F2.softmax(pred_classes, dim = 2)
        max_probability, max_classes = torch.max(pred_classes[0, :, 1:], dim = 1)
        suitable_index = torch.nonzero(max_probability > probability).reshape(-1)
        if suitable_index.numel() == 0:
            return None
        suitable_classes = max_classes[suitable_index] + 1
        suitable_probablity = max_probability[suitable_index]
        decode_anchors = decode_offeset(init_anchors, pred_offeset[0])
        nms_index = []
        for cls in suitable_classes.unique(sorted = False):
            same_index = torch.nonzero(suitable_classes == cls).reshape(-1)
            nms_index.append(nms(
                decode_anchors,
                suitable_probablity[same_index],
                suitable_index[same_index],
                iou_target = nms_iou
            )[0])
        nms_index = torch.cat(nms_index)
        resize_anchors = decode_anchors[nms_index]
        resize_anchors[:, 0] *= self.origin_width
        resize_anchors[:, 1] *= self.origin_height
        resize_anchors[:, 2] *= self.origin_width
        resize_anchors[:, 3] *= self.origin_height
        return (suitable_index, suitable_probablity, suitable_classes, nms_index), resize_anchors

    def imaging(self):
        np_img = F.resize(
            self.image,
            [self.origin_height, self.origin_width],
            antialias = True
        )
        np_img = (np_img.cpu().numpy() * 255).astype(np.uint8)
        np_img = np.transpose(np_img, (1, 2, 0)).copy()
        (suitable_index, suitable_probablity, suitable_classes, nms_index),\
        resize_anchors = self.run(
            self.image, self.origin_width, self.origin_height,
        )
        for index, anchor in enumerate(resize_anchors):
            x1, y1, x2, y2 = [int(value) for value in anchor.detach().cpu().tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.origin_width - 1, x2)
            y2 = min(self.origin_height - 1, y2)
            np_img[y1:y2, x1] = [255, 0, 0]
            np_img[y1:y2, x2] = [255, 0, 0]
            np_img[y1, x1:x2] = [255, 0, 0]
            np_img[y2, x1:x2] = [255, 0, 0]
            anchor_index = nms_index[index]
            origin_index = torch.nonzero(suitable_index == anchor_index).reshape(-1)
            cls = suitable_classes[origin_index].item()
            prob = suitable_probablity[origin_index].item()
            plt.text(x1, y1 - 6, f'{CLASSES[cls - 1]} {prob:.2f}',
                     color = 'red', fontsize = 8,
                     bbox = dict(facecolor = 'white', alpha = 0.7, pad = 0))
        plt.imshow(np_img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    image = read_image(r'./VOC2007测试集/JPEGImages/000001.jpg', ImageReadMode.RGB).float() / 255
    origin_width = image.shape[2]
    origin_height = image.shape[1]
    image = F.resize(image, [300, 300], antialias = True)
    net = SSD()
    checkpoint = torch.load(CHECKPOINT_PATH)
    predict = Predict(net, checkpoint = checkpoint)
    predict.run(image,origin_width,origin_height)
    predict.imaging()

