# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import _init_paths
# import os
# import sys
# import numpy as np
import argparse
# import pprint
# import pdb
# import time
# import cv2
# import torch
from torch.autograd import Variable
# import torch.nn as nn
# import torch.optim as optim
#
# import torchvision.transforms as transforms
# import torchvision.datasets as dset
# from scipy.misc import imread
# from roi_data_layer.roidb import combined_roidb
# from roi_data_layer.roibatchLoader import roibatchLoader
# from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
# from model.rpn.bbox_transform import clip_boxes
# # from model.nms.nms_wrapper import nms
# from model.roi_layers import nms
# from model.rpn.bbox_transform import bbox_transform_inv
# from model.utils.net_utils import save_net, load_net, vis_detections
# from model.utils.blob import im_list_to_blob
# from model.faster_rcnn.vgg16 import vgg16
# from model.faster_rcnn.resnet import resnet
# import pdb
# from loaders.dataset import LadiDatasetMultiInput
# from tqdm import tqdm
# import math
# from classifier.classifier_net import HeadNet
# from classifier.config import multi_task_train_cfg as conf_local
# import classifier.methods_classifier as mth
import torch
import os
from loaders.dataset import CascadeDataset
from classifier.classifier_net import HeadNet, ResNet
import classifier.methods_classifier as mth
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import math
from classifier.config import multi_task_train_cfg as conf_local
from classifier.rcnn_base import _createFasterRCNN, _inferenceFasterRCNN, FasterRCNN_attention


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="/srv/share/jyang375/models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    args = parser.parse_args()
    return args


# ########################################################################################################


def filePath(_prefix, _category):
    return os.path.join(
        conf_local.TASKS_FOLDER,
        _category,
        '{}_{}.json'.format(_prefix, _category))

def cropDataset(_dataset):
    minLength = 0
    for i in _dataset:
        if minLength == 0:
            minLength = len(i)
        else:
            minLength = min(minLength, len(i))

    for i in range(len(_dataset)):
        _diff = len(_dataset[i])-minLength
        if _diff > 0:
            _dataset[i] = _dataset[i][:-_diff]

    return _dataset

# Parameters
args = parse_args()

SESSION = 2
file_no = 32

vanilla = True

notes = 'ResNet101(parts) - adam - lr:0.001 - scheduler 0.5/2'

params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}
MAX_EPOCHS = 400
# split_parameter = 0.9

learning_rate = 0.01
momentum = 0.9

MODEL_FOLDER = os.path.join(conf_local.MODEL_FOLDER, conf_local.CATEGORY)

weights_file = os.path.join(MODEL_FOLDER, 'model_{}_{}.{}.pth'.format(conf_local.CATEGORY, SESSION, file_no))
vanilla_starter = os.path.join(conf_local.MODEL_FOLDER, "pretrained", "resnet101-5d3b4d8f.pth")


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Categories
categories = ["damage", "environment", "infrastructure", "vehicle", "water"]


# Datasets
labels = []
classes = []
trainData = []
testData = []

for _category in categories:
    _labels = mth.readJSON(filePath("labels", _category))
    labels.append(_labels)

    _classes = mth.readJSON(filePath("classes", _category))
    classes.append(_classes)

    _trainData = mth.readJSON(filePath("listIDs_train", _category))
    trainData.append(_trainData)

    _testData = mth.readJSON(filePath("listIDs_test", _category))
    testData.append(_testData)

trainData = cropDataset(trainData)
testData = cropDataset(testData)


# Transformations
tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Loaders
training_set = CascadeDataset(trainData, labels)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = CascadeDataset(testData, labels)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)


# ###########################################################################################
# Model
# ###########################################################################################

net, pascal_classes = _createFasterRCNN(args)
# net, im_data, im_info, num_boxes, gt_boxes, pascal_classes, args = initFasterRCNN()
# net = ResNet()
print("Use device: " + str(device))
net.to(device)
# net.train()

head = [HeadNet(6), HeadNet(8), HeadNet(9), HeadNet(4), HeadNet(5)]

for _net in head:
    _net.to(device)
    _net.train()


# ###########################################################################################
# Loss Function
# ###########################################################################################
loss_function = nn.BCELoss()


# ###########################################################################################
# Optimizer
# ###########################################################################################
headOptimizers = []
headSchedulers = []

# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for _net in head:
    # _optimizer = torch.optim.SGD(_net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)
    _optimizer = torch.optim.Adam(_net.parameters(), lr=learning_rate)
    _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=10, gamma=0.1)
    headOptimizers.append(_optimizer)
    headSchedulers.append(_scheduler)


# ###########################################################################################
# Load State
# Initialize weights
# ###########################################################################################
if vanilla:
    # pretrained_state = torch.load(vanilla_starter)
    # model_state = net.state_dict()
    # pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size()}
    # model_state.update(pretrained_state)
    # net.load_state_dict(model_state)
    start_epoch = 1

else:
    checkpoint = torch.load(weights_file)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for _netId in range(len(head)):
        head[_netId].load_state_dict(checkpoint['head{}'.format(_netId)])
        headOptimizers[_netId].load_state_dict(checkpoint['optimizer{}'.format(_netId)])

    start_epoch = checkpoint['epoch'] + 1



# ###########################################################################################

im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

# ship to cuda

im_data = im_data.cuda()
im_info = im_info.cuda()
num_boxes = num_boxes.cuda()
gt_boxes = gt_boxes.cuda()

# make variable
with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

for param in net.parameters():
    param.requires_grad = False

net.eval()

# ###########################################################################################
# Loop over epochs
# ###########################################################################################
for epoch in range(start_epoch, MAX_EPOCHS):
    # Training
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0



    with tqdm(total=math.ceil(len(training_loader)), desc="Training") as pbar:
        for _inputs, _targets, _, _ in training_loader:
            # scheduler.step()
            # net.train()
            # net.eval()
            # Transfer to GPU
            taskNum = len(_inputs)
            losses = []
            for taskID in range(taskNum):
                inputs = _inputs[taskID]
                targets = _targets[taskID]

                # inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.to(device)
                # Model computations
                # mid = net(inputs)


                mid, _, _, _ = _inferenceFasterRCNN(args, inputs, im_data, im_info, gt_boxes, num_boxes, pascal_classes, net)
                # mid = FasterRCNN_attention(inputs)
                outputs = head[taskID](mid)
                _loss = loss_function(outputs, targets)
                losses.append(_loss)

            # optimizer.zero_grad()
            for taskID in range(taskNum):
                headOptimizers[taskID].zero_grad()
                headSchedulers[taskID].step()

            loss = sum(losses)/5

            loss.backward()
            for taskID in range(taskNum):
                headOptimizers[taskID].step()

            # optimizer.step()

            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss),
                              categories[0]: '{:.2f}'.format(losses[0]),
                              categories[1]: '{:.2f}'.format(losses[1]),
                              categories[2]: '{:.2f}'.format(losses[2]),
                              categories[3]: '{:.2f}'.format(losses[3]),
                              categories[4]: '{:.2f}'.format(losses[4])})
            pbar.update(1)
    pbar.close()


    # Validation
    with torch.set_grad_enabled(False):
        net.eval()

        print('Saving')
        save_name = os.path.join(MODEL_FOLDER, 'model_{}_{}.{}.pth').format(conf_local.CATEGORY, SESSION, epoch)
        mth.save_checkpoint({
            'notes': notes,
            'epoch': epoch,
            'model': net.state_dict(),
            'head0': head[0].state_dict(),
            'head1': head[1].state_dict(),
            'head2': head[2].state_dict(),
            'head3': head[3].state_dict(),
            'head4': head[4].state_dict(),
            # 'optimizer': optimizer.state_dict(),
            'optimizer0': headOptimizers[0].state_dict(),
            'optimizer1': headOptimizers[1].state_dict(),
            'optimizer2': headOptimizers[2].state_dict(),
            'optimizer3': headOptimizers[3].state_dict(),
            'optimizer4': headOptimizers[4].state_dict()
        }, save_name)





# ########################################################################################################
# if __name__ == '__main__':
#     fasterRCNN, im_data, im_info, num_boxes, gt_boxes, pascal_classes, args = initFasterRCNN()
#     imglist = os.listdir(args.image_dir)
#     num_images = len(imglist)
#
#     print('Loaded Photo: {} images.'.format(num_images))
#
#     while (num_images >= 0):
#         num_images -= 1
#         im_file = os.path.join(args.image_dir, imglist[num_images])
#         base_feat, pred_boxes, bb, scores = _inferenceFasterRCNN(args, im_file, im_data, im_info, gt_boxes, num_boxes, pascal_classes, fasterRCNN)
#
#         a=2


