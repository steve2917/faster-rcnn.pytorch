import torch
import argparse
import os
from loaders.dataset import CascadeDataset
from classifier.multitask_classifier_net import HeadNet
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
from sklearn.metrics import average_precision_score
import classifier.methods_classifier as mth
from classifier.config import multi_task_test_cfg as conf_local
from classifier.rcnn_base import _createFasterRCNN, _inferenceFasterRCNN
from torch.autograd import Variable

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
SESSION = 6
file_no = 1


params = {'batch_size': 1,
          'num_workers': 1}


# Initialization
MODEL_FOLDER = os.path.join(conf_local.MODEL_FOLDER, conf_local.CATEGORY)
weights_file = os.path.join(MODEL_FOLDER, 'model_{}_{}.{}.pth'.format(conf_local.CATEGORY, SESSION, file_no))


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Categories
categories = ["damage", "environment", "infrastructure", "vehicle", "water"]

# ###########################################################################################
# Data and Loaders
# ###########################################################################################

# Datasets
labels = []
classes = {'classes': []}
testData = []

for _category in categories:
    _labels = mth.readJSON(filePath("labels", _category))
    labels.append(_labels)

    _classes = mth.readJSON(filePath("classes", _category))
    classes['classes'].extend(_classes['classes'])

    _testData = mth.readJSON(filePath("listIDs_test", _category))
    testData.append(_testData)

for i in range(len(classes['classes'])):
    classes['classes'][i]['id'] = i

testData = cropDataset(testData)


# Transformations
tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Loaders
validation_set = CascadeDataset(testData, labels)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)


# ###########################################################################################
# Model
# ###########################################################################################

# backbone network (ResNet101)
net, pascal_classes = _createFasterRCNN(args)
print("Use device: " + str(device))
net.to(device)
net.eval()

# Heads networks for every task
head = [HeadNet(6), HeadNet(8), HeadNet(9), HeadNet(4), HeadNet(5)]

for _net in head:
    _net.to(device)
    _net.eval()


# ###########################################################################################
# Load State
# ###########################################################################################

# Load .pth file
print('Loading: {}'.format(weights_file))
checkpoint = torch.load(weights_file)

# Load backbone Weights
net.load_state_dict(checkpoint['model'])

# Load head Weights
for _netId in range(len(head)):
    head[_netId].load_state_dict(checkpoint['head{}'.format(_netId)])


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
# Inference over Test Data
# ###########################################################################################
with torch.set_grad_enabled(False):
    # total = 0
    # cm = np.zeros((class_num, 2, 2))
    flag_first_batch = True
    with tqdm(total=math.ceil(len(validation_loader)), desc="Testing") as pbar:
        for _inputs, _targets, _, _ in validation_loader:
            taskNum = len(_inputs)

            for taskID in range(taskNum):
                inputs = _inputs[taskID]
                targets = _targets[taskID]

                # inputs = inputs.to(device)
                # inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.to(device)

                base_feat, _,  _, _ = _inferenceFasterRCNN(args, inputs, im_data, im_info, gt_boxes, num_boxes, pascal_classes, net)
                _outputs = head[taskID](base_feat)

                if taskID == 0:
                    _outputs_merged = _outputs.cpu()
                    _targets_merged = targets.cpu()
                else:
                    _outputs_merged = np.hstack((_outputs_merged, _outputs.cpu()))
                    _targets_merged = np.hstack((_targets_merged, targets.cpu()))

            if flag_first_batch:
                outputs = _outputs_merged
                targ = _targets_merged
                flag_first_batch = False

            else:
                outputs = np.vstack((outputs, _outputs_merged))
                targ = np.vstack((targ, _targets_merged))
            pbar.update(1)
    pbar.close()

    ap = average_precision_score(targ, outputs, average=None)
    map = average_precision_score(targ, outputs)
    mth.display_map(ap, map, classes,  conf_local.CATEGORY, total=0, notes='notes',  epoch=0, session=0)
