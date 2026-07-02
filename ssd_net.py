import torch
from torch import nn

from anchors_opt import creat_anchors
from config import SIZE_RATE, W_H_RATE


def vgg_block(num_conv: int, in_channels: int, out_channels: int,
              pool_kernel = 2, pool_stride = 2, pool_padding = 0,
              is_pool = True) -> Sequential:
    layers = []
    for _ in range(num_conv):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    if is_pool:
        layers.append(nn.MaxPool2d(
            kernel_size = pool_kernel,
            stride = pool_stride,
            padding = pool_padding
        ))
    return nn.Sequential(*layers)


def dilate_block(in_channels: int, out_channels: int,
                 dilate_padding = 6, dilation = 6) -> Sequential:
    return nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            vgg_block(
                num_conv =  3,
                in_channels = in_channels,
                out_channels = in_channels,
                pool_kernel = 3,
                pool_stride = 1,
                pool_padding = 1
            ),
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                padding = dilate_padding,
                dilation = dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


def assist_block(in_channels: int, out_channels: int,
                 stride = 2, padding = 1) -> Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels//2,
            kernel_size = 1,
        ),
        nn.BatchNorm2d(out_channels//2),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = out_channels//2,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            padding = padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def detect_classes(num_classes: int, num_anchors: int, in_channels: int) -> Module:
    out_channels = num_anchors * (num_classes + 1)
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)


def detect_anchor(num_anchors: int, in_channels: int) -> Module:
    out_channels = num_anchors * 4
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)


def base_net(in_channels: List, num_conv: List) -> Sequential:
    vgg_blocks = []
    in_channels = in_channels
    num_conv = num_conv
    for i in range(len(in_channels) - 1):
        is_pool = True
        if i == len(in_channels) - 2:
            is_pool = False
        vgg_blocks.append(vgg_block(
            num_conv =  num_conv[i],
            in_channels = in_channels[i],
            out_channels = in_channels[i + 1],
            is_pool = is_pool
        ))
    return nn.Sequential(*vgg_blocks)


def get_blk(index: int) -> Sequential:
    if index == 0:
        in_channels = [3, 64, 128, 256, 512]
        num_conv = [2, 2, 3, 3]
        blk = base_net(in_channels = in_channels, num_conv = num_conv)
    elif index == 1:
        blk = dilate_block(
            in_channels = 512,
            out_channels= 1024,
            dilate_padding = 6,
            dilation = 6
        )
    elif index == 2:
        blk = assist_block(
            in_channels = 1024,
            out_channels = 512,
            stride = 2,
            padding = 1
        )
    elif index == 3:
        blk = assist_block(
            in_channels = 512,
            out_channels = 256,
            stride = 2,
            padding = 1
        )
    elif index in (4,5):
        blk = assist_block(
            in_channels = 256,
            out_channels = 256,
            stride = 1,
            padding = 0
        )
    return blk


def blk_forward(X: Tensor, size_rate: List, w_h_rate: List, blk: Sequential,
                detect_classes: Module, detect_anchor: Module) -> tuple:
    output = blk(X)
    init_anchors = creat_anchors(
        X = output, size_rate = size_rate, w_h_rate = w_h_rate
    )
    classes = detect_classes(output)
    anchors = detect_anchor(output)
    return output, init_anchors, classes, anchors


def flatten_tensor(tensor) -> Tensor:
    return torch.flatten(tensor.permute(0, 2, 3, 1), start_dim = 1)


class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 20
        out_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = [3, 5, 5, 5, 3, 3]
        for index in range(6):
            setattr(self, f"blk_{index}", get_blk(index))
            setattr(self, f"classes_{index}", detect_classes(
                num_classes = self.num_classes,
                num_anchors = num_anchors[index],
                in_channels = out_channels[index]
            ))
            setattr(self, f"anchors_{index}", detect_anchor(
                num_anchors = num_anchors[index],
                in_channels = out_channels[index]

            ))

    def forward(self, X: Tensor) -> tuple:
        init_anchors = [None] * 6
        anchors = [None] * 6
        classes = [None] * 6
        for index in range(6):
            output, init_anchors[index], classes[index], anchors[index] = blk_forward(
                X = X,
                size_rate = SIZE_RATE[index],
                w_h_rate = W_H_RATE[index],
                blk = getattr(self, f"blk_{index}"),
                detect_anchor = getattr(self, f"anchors_{index}"),
                detect_classes = getattr(self, f"classes_{index}")
            )
            X = output
        init_anchors = torch.cat(init_anchors, dim = 0)
        classes = torch.cat([flatten_tensor(item) for item in classes], dim = 1)
        classes = classes.reshape(classes.size(0), -1, self.num_classes+1)
        anchors = torch.cat([flatten_tensor(item) for item in anchors], dim = 1)
        anchors = anchors.reshape(anchors.size(0), -1, 4)
        # (样本数, 检测头数*锚框数, 4)
        # (样本数, 检测头数*锚框数*长*宽*类别数)
        # (样本数, 检测头数*锚框数*4)
        return init_anchors, classes, anchors








