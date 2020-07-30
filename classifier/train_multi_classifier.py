# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import classifier.methods_classifier as mth
from loaders.dataset import LadiDataset
from classifier.multitask_classifier_net import HeadNet
from tqdm import tqdm
import math
from sklearn.metrics import multilabel_confusion_matrix, average_precision_score
from classifier.config import multi_classifier_train_cfg as conf_local

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


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


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _createFasterRCNN(args):
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'Pipe', 'Car',
                                 'Highway', 'Aircraft',
                                 'Building', 'Railway',
                                 'Rooftop', 'Train',
                                 'Buildings_block', 'Dam / levee',
                                 'Bridge', 'Power_line',
                                 'Boat', 'Dock',
                                 'Road', 'Container_building',
                                 'Communication_tower', 'Truck', 'Airway',
                                 'Water_tower', 'Parking_lot'])

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))

    return fasterRCNN, pascal_classes


def _initializeFasterRCNN(args, fasterRCNN):
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
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

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    return fasterRCNN, im_data, im_info, num_boxes, gt_boxes


def _prepareImage(im_file, im_data, im_info, gt_boxes, num_boxes ):
    im_in = np.array(imread(im_file))
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :, ::-1]

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

    return im_data, im_info, gt_boxes, num_boxes, im_scales


def _inferenceFasterRCNN(args, im_file, im_data, im_info, gt_boxes, num_boxes, pascal_classes, fasterRCNN):

    im_data, im_info, gt_boxes, num_boxes, im_scales = _prepareImage(im_file, im_data, im_info, gt_boxes, num_boxes)
    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, base_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # pred_boxes /= im_scales[0]
    bb = ((pred_boxes[0].cpu().numpy()) / 16).astype(np.int)

    # Construct Mask
    mask = base_feat.clone()[0][0] * 0
    for i in range(len(bb)):
        m_score = max(scores[0][i][1:])
        if m_score > 0.5:
            mask[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] += m_score

    a = mask.max()
    mask /= a
    # Apply Mask
    for i in range(len(base_feat[0])):
        base_feat[0][i] = base_feat[0][i] * (1 + mask)

    return base_feat, pred_boxes, bb, scores


def initFasterRCNN():
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    fasterRCNN, pascal_classes = _createFasterRCNN(args)
    fasterRCNN, im_data, im_info, num_boxes, gt_boxes = _initializeFasterRCNN(args, fasterRCNN)
    fasterRCNN.eval()

    return fasterRCNN, im_data, im_info, num_boxes, gt_boxes, pascal_classes, args


def FasterRCNN_attention(im_file):
    fasterRCNN, im_data, im_info, num_boxes, gt_boxes, pascal_classes, args = initFasterRCNN()
    base_feat, _, _, _ = _inferenceFasterRCNN(args, im_file, im_data, im_info, gt_boxes, num_boxes, pascal_classes, fasterRCNN)
    return base_feat
# ########################################################################################################



# ###########################################################################################
# Parameters
# ###########################################################################################
SESSION = 99
file_no = 6

CLASS_NUM = 9
vanilla = True

notes = 'ResNet101(parts) - adam - lr:0.001 - scheduler 0.5/2'


params = {'batch_size': 70,
          'shuffle': True,
          'num_workers': 20}
MAX_EPOCHS = 4
split_parameter = 0.9

learning_rate = 0.001
momentum = 0.9

MODEL_FOLDER = os.path.join(conf_local.MODEL_FOLDER, conf_local.CATEGORY)

weights_file = os.path.join(MODEL_FOLDER, 'model_{}_{}.{}.pth'.format(conf_local.CATEGORY, SESSION, file_no))
#vanilla_starter = os.path.join("models/pretrained", "resnet50-19c8e357.pth")
vanilla_starter = os.path.join(conf_local.MODEL_FOLDER, "pretrained", "resnet101-5d3b4d8f.pth")


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# ###########################################################################################
# Data and Loaders
# ###########################################################################################

# Datasets
labels = mth.readJSON(conf_local.LABELS_FILE)
classes = mth.readJSON(conf_local.CLASSES_FILE)

trainData = mth.readJSON(conf_local.LIST_IDS_TRAIN_FILE)
testData = mth.readJSON(conf_local.LIST_IDS_TEST_FILE)

# Transformations
tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loaders
training_set = LadiDataset(trainData, labels, tfms)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = LadiDataset(testData, labels, tfms)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)


# ###########################################################################################
# Model - Loss Function - Optimizer
# ###########################################################################################

# Model
net = ResNet(CLASS_NUM)
print("Use device: " + str(device))
net.to(device)
net.train()

# Loss Function
#loss_function = nn.BCEWithLogitsLoss()
loss_function = nn.BCELoss()

# Optimizer
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)


# ###########################################################################################
# Load State
# ###########################################################################################

if vanilla:
    pretrained_state = torch.load(vanilla_starter)
    model_state = net.state_dict()
    pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size()}
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)
    start_epoch = 1

else:
    checkpoint = torch.load(weights_file)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1


# ###########################################################################################
# Loop over epochs
# ###########################################################################################
for epoch in range(start_epoch, MAX_EPOCHS):
    # Training
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    with tqdm(total=math.ceil(len(training_loader)), desc="Training") as pbar:
        for inputs, targets in training_loader:
            scheduler.step()
            net.train()
            # Transfer to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Model computations
            scheduler.step()
            outputs = net(inputs)
            loss = loss_function(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss_function(outputs, targets).item()
            predicted = torch.round(outputs.data)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum()

            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss), 'accurate': '{:.2%}'.format(correct.item() / (total*CLASS_NUM))})
            pbar.update(1)
    pbar.close()


    # Validation
    with torch.set_grad_enabled(False):
        net.eval()

        print('Saving')
        save_name = os.path.join (MODEL_FOLDER, 'model_{}_{}.{}.pth').format(conf_local.CATEGORY, SESSION, epoch)
        mth.save_checkpoint({
          'notes': notes,
          'class_num': CLASS_NUM,
          'epoch': epoch,
          'model': net.state_dict(),
          'optimizer': optimizer.state_dict()
        }, save_name)

        total = 0
        cm = np.zeros((CLASS_NUM, 2, 2))
        flag_first_batch = True
        with tqdm(total=math.ceil(len(validation_loader)), desc="Testing") as pbar:
            for inputs, faster, targets in validation_loader:
                inputs, faster, targets = inputs.to(device), faster.to(device), targets.to(device)
                _, outputs = net(inputs, faster)
                predicted = torch.round(outputs.data)
                total += targets.size(0)
                cm += multilabel_confusion_matrix(targets.cpu(), predicted.cpu())
                if flag_first_batch:
                    out = outputs.cpu()
                    targ = targets.cpu()
                    flag_first_batch = False
                else:
                    out = np.vstack((out, outputs.cpu()))
                    targ = np.vstack((targ, targets.cpu()))
                pbar.update(1)
        pbar.close()
        ap = average_precision_score(targ, out, average=None)
        map = average_precision_score(targ, out)
        mth.display_table(cm, ap, map, classes, conf_local.CATEGORY, total, notes, epoch, SESSION)


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


