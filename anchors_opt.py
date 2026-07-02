import torch


def to_center(anchors: Tensor) -> Tensor:
    x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    center_anchors = torch.stack((center_x, center_y, width, height), dim = 1)
    return center_anchors


def to_corner(anchors: Tensor) -> Tensor:
    center_x, center_y, width, height = \
        anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    x1 = center_x - 0.5 * width
    y1 = center_y - 0.5 * height
    x2 = center_x + 0.5 * width
    y2 = center_y + 0.5 * height
    corner_anchors = torch.stack((x1, y1, x2, y2), dim = 1)
    return corner_anchors


def creat_anchors(X: Tensor, size_rate: List, w_h_rate: List) -> Tensor:
    height, width = X.shape[-2:]
    device, num_size, num_w_h = X.device, len(size_rate), len(w_h_rate)
    num_anchors = num_size + num_w_h - 1
    size_tensor = torch.tensor(size_rate, device = device)
    w_h_tensor = torch.tensor(w_h_rate, device = device)

    offset_h, offset_w = 0.5, 0.5
    h_center = (torch.arange(height, device = device) + offset_h) / height
    w_center = (torch.arange(width, device = device) + offset_w) / width
    h_center = h_center.repeat_interleave(width)
    w_center = w_center.repeat(height)

    w = torch.cat((size_tensor * torch.sqrt(w_h_tensor[0]),
                  size_rate[0] * torch.sqrt(w_h_tensor[1:])))
    h = torch.cat((size_tensor / torch.sqrt(w_h_tensor[0]),
                   size_rate[0] / torch.sqrt(w_h_tensor[1:])))

    anchor_size = torch.stack(
        (-w, -h, w, h), dim=1
    ).repeat(width * height, 1) / 2
    anchor_center = torch.stack(
        (w_center, h_center, w_center, h_center), dim=1
    ).repeat_interleave(num_anchors, dim = 0)

    anchors = anchor_center + anchor_size
    return anchors


def anchors_iou(anchor1: Tensor, anchor2: Tensor) -> Tensor:
    def anchor_area(anchor) -> float:
        return (anchor[:, 2] - anchor[:, 0]) * (anchor[:, 3] - anchor[:, 1])
    anchor1_area = anchor_area(anchor1)
    anchor2_area = anchor_area(anchor2)
    intersect_topleft = torch.max(anchor1[:, :2], anchor2[:, None, :2])
    intersect_bottumright = torch.min(anchor1[:, 2:], anchor2[:, None, 2:])
    intersect_anchor = (intersect_bottumright - intersect_topleft).clamp(min = 0)
    intersect_area = intersect_anchor[:, :, 0] * intersect_anchor[:, :, 1]
    union_area = anchor1_area + anchor2_area[:, None] - intersect_area
    iou_tensor = intersect_area / union_area
    return iou_tensor


def allocate_anchors(init_anchors: Tensor, true_anchors: Tensor, target = 0.5) -> Tensor:
    num_Ianchors = init_anchors.shape[0]
    num_Tanchors = true_anchors.shape[0]
    iou_tensor = anchors_iou(true_anchors, init_anchors)
    allocate_map = torch.full(
        (num_Ianchors,), -1, device = init_anchors.device
    )
    width_max, width_indices = torch.max(iou_tensor, dim = 1)
    Ianchors_indices = torch.nonzero(width_max >= target).reshape(-1)
    Tanchors_indices = width_indices[width_max >= target]
    allocate_map[Ianchors_indices] = Tanchors_indices
    for _ in range(num_Tanchors):
        Tanchors_indices = torch.argmax(iou_tensor) % num_Tanchors
        Ianchors_indices = torch.argmax(iou_tensor) // num_Tanchors
        allocate_map[Ianchors_indices] = Tanchors_indices
        iou_tensor[:, Tanchors_indices] = torch.full((num_Ianchors,), -1)
        iou_tensor[Ianchors_indices, :] = torch.full((num_Tanchors,), -1)
    return allocate_map


def encode_offeset(init_anchors: Tensor, true_anchors: Tensor, eps = 1e-6) -> Tensor:
    center_Ianchors = to_center(init_anchors)
    center_Tanchors = to_center(true_anchors)
    offeset_xy = 10 * (center_Tanchors[:, :2] - center_Ianchors[:, :2]) / (center_Ianchors[:, 2:] + eps)
    offeset_wh = 5 * torch.log(center_Tanchors[:, 2:] / center_Ianchors[:, 2:] + eps)
    offeset = torch.cat((offeset_xy, offeset_wh), dim = 1)
    return offeset


def decode_offeset(init_anchors: Tensor, offeset: Tensor) -> Tensor:
    center_Ianchors = to_center(init_anchors)
    decode_xy = offeset[:, :2] * center_Ianchors[:, 2:] / 10 + center_Ianchors[:, :2]
    decode_wh = torch.exp(offeset[:, 2:] / 5) * center_Ianchors[:, 2:]
    decode_anchors = torch.cat((decode_xy, decode_wh), dim = 1)
    return to_corner(decode_anchors)


def sign_anchors(init_anchors: Tensor, lables: List) -> tuple:
    batch_size = len(lables)
    num_anchors = init_anchors.shape[0]
    batch_classes, batch_offeset, batch_mask = [], [], []
    for i in range(batch_size):
        lable = lables[i]
        allocate_map = allocate_anchors(init_anchors, lable[: , 1:])
        classes = torch.zeros(
            num_anchors, device = init_anchors.device, dtype = torch.long
        )
        true_anchors = torch.zeros(
            (num_anchors, 4), device = init_anchors.device
        )
        allocate_Iindices = torch.nonzero(allocate_map >= 0).reshape(-1)
        allocate_Tindices = allocate_map[allocate_Iindices]
        classes[allocate_Iindices] = (lable[allocate_Tindices, 0]).long()
        true_anchors[allocate_Iindices] = lable[allocate_Tindices, 1:]
        offeset = encode_offeset(init_anchors, true_anchors)
        allocate_mask = ((allocate_map >= 0).unsqueeze(-1)).repeat(1, 4)
        offeset_mask = offeset * allocate_mask
        batch_classes.append(classes.reshape(-1))
        batch_offeset.append(offeset_mask.reshape(-1))
        batch_mask.append(allocate_mask.reshape(-1))
    batch_classes = torch.stack(batch_classes)
    batch_offeset = torch.stack(batch_offeset)
    batch_mask = torch.stack(batch_mask)
    # (样本数, 锚框数*1)
    # (样本数, 锚框数*4)
    # (样本数, 锚框数*4)
    return batch_classes, batch_offeset, batch_mask


def nms(decode_anchors: Tensor, probablity_list: List,
        anchors_index: List, iou_target = 0.45) -> tuple:
    live_list = []
    live_probablity = []
    probablity_copylist = probablity_list.clone()
    anchors_copyindex = anchors_index.clone()
    while probablity_copylist.numel() > 0:
        max_probablity, max_indices = torch.max(probablity_copylist, dim = -1)
        live_list.append(anchors_copyindex[max_indices])
        live_probablity.append(max_probablity)
        if probablity_copylist.numel() == 1:
            break
        probablity_copylist = torch.cat(
            (probablity_copylist[:max_indices],
             probablity_copylist[max_indices + 1:])
        )
        anchors_copyindex = torch.cat(
            (anchors_copyindex[:max_indices],
             anchors_copyindex[max_indices + 1:])
        )
        iou = anchors_iou(
            decode_anchors[live_list[-1], :].unsqueeze(0),
            decode_anchors[anchors_copyindex, :]
        ).reshape(-1)
        live_index = torch.nonzero(iou <= iou_target).reshape(-1)
        anchors_copyindex = anchors_copyindex[live_index]
        probablity_copylist = probablity_copylist[live_index]
    live_list = torch.stack(live_list).detach()
    live_probablity = torch.stack(live_probablity).detach()
    return live_list, live_probablity


