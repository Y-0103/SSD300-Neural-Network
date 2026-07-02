# 类型
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 模型
IMAGE_SIZE = 300
SIZE_RATE = [[0.1],
             [0.2],
             [0.37],
             [0.54],
             [0.71],
             [0.88]]
W_H_RATE = [[1, 2, 0.5],
            [1, 2, 0.5, 3, 0.33],
            [1, 2, 0.5, 3, 0.33],
            [1, 2, 0.5, 3, 0.33],
            [1, 2, 0.5],
            [1, 2, 0.5]]

# 训练
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCH = 201
LOG_PATH = r'.\train_log.txt'
PARAMS_SAVE_PATH = r'.\checkpoint'
FILE_PATH = r'.\VOC2007+2012训练集'

# 测试
NMS_IOU = 0.45
HIT_IOU = 0.5
CONF_THRESH = 0.01
CHECKPOINT_PATH = r'.\checkpoint\ssd_checkpoint_epoch_175.pth'
TEST_FILE_PATH = r'.\VOC2007测试集'

