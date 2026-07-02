import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image, ImageReadMode

from pathlib import Path
from collections import defaultdict

import config
from ssd_net import SSD
from anchors_opt import anchors_iou
from parse_data import save_data
from ssd_predict import Predict


class Test_Dataset:
    def __init__(self, file_path: str):
        self.cache_data = None
        self.file_path = file_path
        self.image_size = config.IMAGE_SIZE
        self.load_data(file_path)

    def load_data(self, file_path: str, cache_path = 'test_annotations.pt'):
        file_path = Path(file_path)
        cache_file = file_path / cache_path
        if not cache_file.exists():
            save_data(file_path, cache_path)
        self.cache_data = torch.load(cache_file, weights_only = False)

    def __len__(self):
        return len(self.cache_data['names'])

    def __getitem__(self, idx) -> tuple:
        file_path = Path(self.file_path)
        images_file = file_path / 'JPEGImages'
        image_path = images_file / (self.cache_data['names'][idx] + '.jpg')
        image = read_image(str(image_path), ImageReadMode.RGB).float() / 255
        image = F.resize(image, [self.image_size, self.image_size], antialias = True)
        classes_anchors = self.cache_data['classes_anchors'][idx].clone()
        return image, classes_anchors


class Test:
    def __init__(self, file_path: str, net: Modules, checkpoint: dict):
        self.net = net
        self.checkpoint = checkpoint
        self.dataset = Test_Dataset(file_path)
        self.cache_data = self.dataset.cache_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = config.CONF_THRESH
        self.nms_iou = config.NMS_IOU
        self.predict = Predict(self.net, self.checkpoint)

    def predict_data(self):
        num_samples = len(self.cache_data['names'])
        predict_result = defaultdict(list)
        resize_Tanchors = defaultdict(list)
        for sample in range(num_samples):
            image_name = self.cache_data['names'][sample]
            width, height = self.cache_data['sizes'][sample]
            image, classes_anchors = self.dataset[sample]
            result = self.predict.run(
                image, width, height,
                probability = self.conf_thresh,
                nms_iou = self.nms_iou
            )
            if result is None:
                continue
            (suitable_index, suitable_probablity, suitable_classes, nms_index), \
            resize_anchors = result
            for index in range(nms_index.numel()):
                anchor_idx = nms_index[index]
                origin_idx = torch.nonzero(suitable_index == anchor_idx).reshape(-1)
                cls = suitable_classes[origin_idx].item()
                conf = suitable_probablity[origin_idx].item()
                x1 = resize_anchors[index, 0].item()
                y1 = resize_anchors[index, 1].item()
                x2 = resize_anchors[index, 2].item()
                y2 = resize_anchors[index, 3].item()
                predict_result[image_name].append(
                    [cls, conf, x1, y1, x2, y2]
                )
            for anchor in classes_anchors:
                true_cls = anchor[0].item()
                true_x1 = anchor[1].item() * width
                true_y1 = anchor[2].item() * height
                true_x2 = anchor[3].item() * width
                true_y2 = anchor[4].item() * height
                resize_Tanchors[image_name].append([
                    true_cls, true_x1, true_y1, true_x2, true_y2
                ])
        return predict_result, resize_Tanchors

    def calculate_mAP(self):
        predict_result, resize_Tanchors = self.predict_data()
        print(f"图片数: {len(resize_Tanchors)}")
        num_anchors = sum(len(value) for value in predict_result.values())
        print(f"锚框数: {num_anchors} ")
        aps = {}
        for cls_id in range(1, 21):
            cls_name = config.CLASSES[cls_id - 1]
            cls_Ianchors = []
            for file_name, Ianchors in predict_result.items():
                for cls, conf, x1, y1, x2, y2 in Ianchors:
                    if cls == cls_id:
                        cls_Ianchors.append((file_name, conf, x1, y1, x2, y2))
            cls_Ianchors.sort(key = lambda x: x[1], reverse = True)
            cls_Tanchors = {}
            for file_name, Tanchors in resize_Tanchors.items():
                true_anchors = [Tanchor[1:] for Tanchor in Tanchors if Tanchor[0] == cls_id]
                if true_anchors:
                    cls_Tanchors[file_name] = true_anchors
            matched = {
                file_name: [False] * len(Tanchors) \
                for file_name, Tanchors in cls_Tanchors.items()
            }
            TP = torch.zeros(len(cls_Ianchors))
            FP = torch.zeros(len(cls_Ianchors))
            for i, (file_name, conf, x1, y1, x2, y2) in enumerate(cls_Ianchors):
                if file_name not in cls_Tanchors:
                    FP[i] = 1
                    continue
                true_anchors = torch.tensor(cls_Tanchors[file_name])
                pred_anchor = torch.tensor([[x1, y1, x2, y2]])
                iou = anchors_iou(true_anchors, pred_anchor)
                max_iou, max_indices = iou.max(dim = -1)
                if max_iou >= config.HIT_IOU and not matched[file_name][max_indices.item()]:
                    TP[i] = 1
                    matched[file_name][max_indices.item()] = True
                else:
                    FP[i] = 1

            cum_tp = TP.cumsum(dim = 0)
            cum_fp = FP.cumsum(dim = 0)
            npos = sum(len(value) for value in cls_Tanchors.values())
            recall = cum_tp / npos
            precision = cum_tp / (cum_tp + cum_fp).clamp(min = 1e-12)
            ap = self.AP_area(recall, precision)
            aps[cls_name] = ap
            print(f" {cls_name:>14s}  AP = {ap:.4f}")

        mAP = sum(aps.values()) / len(aps) if aps else 0.0
        print(f"\n{'=' * 40}")
        print(f"  mAP = {mAP:.4f}")
        print(f"{'=' * 40}")
        return mAP

    @staticmethod
    def AP_11(recall, precision):
        ap = 0.0
        for i in torch.linspace(0, 1, 11):
            mask = recall >= i
            if mask.any():
                ap += precision[mask].max().item()
        return ap / 11.0

    @staticmethod
    def AP_area(recall, precision):
        recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
        precision = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = torch.max(precision[i], precision[i + 1])
        mask = recall[1:] != recall[:-1]
        ap = ((recall[1:] - recall[:-1])[mask] *precision[1:][mask]).sum().item()
        return ap


# ==================== 入口 ====================
if __name__ == '__main__':
    net = SSD()
    checkpoint = torch.load(config.CHECKPOINT_PATH)
    test = Test(config.TEST_FILE_PATH, net, checkpoint)
    test.calculate_mAP()












