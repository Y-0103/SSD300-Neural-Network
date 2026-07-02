import torch
from torch import nn
import logging, sys

import config
from anchors_opt import sign_anchors
from load_data import Voc_Dataset, collate_voc
from ssd_net import SSD


def params_split(net: Modules) -> tuple:
    bn_params = []
    other_params = []
    bn_param_ids = set()
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                bn_param_ids.add(id(param))
                if param.requires_grad:
                    bn_params.append(param)
    for param in net.parameters():
        if param.requires_grad and id(param) not in bn_param_ids:
            other_params.append(param)
    return bn_params, other_params


def all_loss(pred_classes: Tensor, sign_classes: Tensor, pred_offeset: Tensor,
             sign_offeset: Tensor, allocate_masks: Tensor, npos_rate: int) -> Tensor:
    batch_size = pred_classes.shape[0]
    num_classes = pred_classes.shape[2]
    cl_loss = classes_loss(
        pred_classes.reshape(-1, num_classes),
        sign_classes.reshape(-1)
    ).reshape(batch_size, -1)
    final_loss = torch.zeros(batch_size, device = pred_classes.device)
    for batch in range(batch_size):
        pos_mas = sign_classes[batch] > 0
        neg_mas = sign_classes[batch] == 0
        num_pos = pos_mas.sum()
        num_neg = neg_mas.sum()
        k = min(num_pos * npos_rate, num_neg)
        pos_loss = 0
        if num_pos == 0:
            k = 2
        else:
            pos_loss = cl_loss[batch][pos_mas].mean()
        neg_loss = cl_loss[batch][neg_mas].topk(k).values.mean()
        f_loss = pos_loss + npos_rate * neg_loss
        final_loss[batch] = f_loss

    num_mask = (allocate_masks.sum(dim = 1) / 4).clamp(min = 1)
    off_loss = offeset_loss(
        pred_offeset.reshape(-1, 4) * allocate_masks.reshape(-1, 4),
        sign_offeset.reshape(-1,4) * allocate_masks.reshape(-1, 4)
    ).reshape(batch_size, -1).sum(dim = 1)/ num_mask

    return final_loss + off_loss


def train(net: Modules, num_epochs: int, data_iter: DataLoader, device,
          sgd: Optimizer, scheduler: MultiStepLR) -> tuple:
    for epoch in range(num_epochs):
        net.train()
        for images, clsses_anchors in data_iter:
            sgd.zero_grad()
            X, lables = images.to(device), clsses_anchors
            lables = [l.to(device) for l in lables]
            init_anchors, pred_classes, pred_offeset = net(X)
            sign_classes, sign_offeset, batch_masks = sign_anchors(init_anchors, lables)
            npos_rate = 1 if epoch <= 35 else 2 if epoch <= 70 else 3
            loss = all_loss(
                pred_classes, sign_classes,
                pred_offeset, sign_offeset, batch_masks, npos_rate
            )
            loss.mean().backward()
            sgd.step()
            logging.info(f"epoch：{epoch}；loss：{loss.mean().item():.4f}")

        scheduler.step()
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': sgd.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(
                checkpoint,
                config.PARAMS_SAVE_PATH + f'/checkpoint_epoch_{epoch}.pth'
            )


if __name__ == '__main__':
    voc_dataset = Voc_Dataset(file_path = config.FILE_PATH)
    data_iter = torch.utils.data.DataLoader(
        voc_dataset, num_workers = config.NUM_WORKERS,
        batch_size = config.BATCH_SIZE, shuffle = True, collate_fn = collate_voc
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SSD()
    net = net.to(device)
    num_epochs = config.EPOCH
    bn_params, other_params = params_split(net)
    sgd = torch.optim.SGD([
        {'params': other_params, 'lr': 0.001, 'weight_decay': 5e-4},
        {'params': bn_params, 'lr': 0.001, 'weight_decay': 0.0}
    ], momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        sgd, milestones = [105, 150], gamma = 0.1
    )
    classes_loss = nn.CrossEntropyLoss(reduction = 'none')
    offeset_loss = nn.SmoothL1Loss(reduction = 'none')
    logging.basicConfig(
        level = logging.INFO,
        format = '%(message)s',
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                config.LOG_PATH, encoding = 'utf-8'
            )]
    )
    # 开始训练
    train(
        net = net,
        num_epochs = num_epochs,
        data_iter = data_iter,
        device = device,
        sgd = sgd,
        scheduler = scheduler
    )




