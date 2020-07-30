import torch
import os
import argparse
from loaders.dataset import LadiDatasetInference
from classifier.multitask_classifier_net import HeadNet
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
import time
import classifier.methods_classifier as mth
from classifier.config import multi_task_test_cfg as conf_local
from classifier.rcnn_base import _createFasterRCNN, _inferenceFasterRCNN
import json
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


def removeDuplicates(_list):
    _list2 = []
    for _item in _list:
        _item = _item.split('_')[0] + "_" + _item.split('_')[1]
        if _item not in _list2:
            _list2.append(_item)

    return _list2


def run(testData, args,  _work_dir='/home/demertzis/GitHub/faster-rcnn.pytorch/tmp'):
    # Parameters
    # args = parse_args()

    SESSION = 6
    file_no = 1

    params = {'batch_size': 100,
              'num_workers': 1}

    # Initialization
    MODEL_FOLDER = os.path.join(conf_local.MODEL_FOLDER, conf_local.CATEGORY)
    weights_file = os.path.join(MODEL_FOLDER, 'model_{}_{}.{}.pth'.format(conf_local.CATEGORY, SESSION, file_no))
    # fasterRcnn_file = './models/res101/ladi/faster_rcnn_1_120_283.pth'


    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Categories
    categories = ["damage", "environment", "infrastructure", "vehicle", "water"]
    # sessions = [7, 1, 99, 1, 1]
    # file_nums = [3, 3, 1, 3, 3]

    # ###########################################################################################
    # Model
    # ###########################################################################################

    # backbone network (ResNet101)
    fasterRCNN, pascal_classes = _createFasterRCNN(args)
    print("Use device: " + str(device))
    fasterRCNN.to(device)
    fasterRCNN.eval()

    # Heads and Base networks for every task
    head = [HeadNet(6), HeadNet(8), HeadNet(9), HeadNet(4), HeadNet(5)]

    for _net in head:
        _net.to(device)
        _net.eval()


    # ###########################################################################################
    # Data and Loaders
    # ###########################################################################################

    # Datasets
    classes = {'classes': []}

    for _category in categories:
        _classes = mth.readJSON(filePath("classes", _category))
        classes['classes'].extend(_classes['classes'])

    for i in range(len(classes['classes'])):
        classes['classes'][i]['id'] = i

    # testData = mth.readJSON(LIST_IDS_INFERENCE_FILE)

    # Transformations
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Loaders
    validation_set = LadiDatasetInference(testData, transforms=tfms)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params)


    # ###########################################################################################
    # Load State
    # ###########################################################################################

    # Load .pth file
    print('Loading: {}'.format(weights_file))
    checkpoint = torch.load(weights_file)
    # checkpoint_faster = torch.load(fasterRcnn_file)

    # Load backbone Weights
    fasterRCNN.load_state_dict(checkpoint['model'])

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

    for param in fasterRCNN.parameters():
        param.requires_grad = False

    fasterRCNN.eval()

    # ###########################################################################################
    # Inference over Test Data
    # ###########################################################################################
    with torch.set_grad_enabled(False):
        flag_first_batch = True
        start_time = time.time()
        with tqdm(total=math.ceil(len(validation_loader)), desc="Testing") as pbar:
            # for _inputs, _targets, _uuids in validation_loader:
            for _filenames, _uuids, _, _, _ in validation_loader:
                taskNum = len(head)

                # mask = _createMask(args, _filenames, im_data, im_info, gt_boxes, num_boxes, pascal_classes, fasterRCNN)
                base_feat, _, _, _ = _inferenceFasterRCNN(args, _filenames, im_data, im_info, gt_boxes, num_boxes,
                                                          pascal_classes, fasterRCNN)
                for taskID in range(taskNum):
                    # inputs = inputs.to(device)

                    # output1 = base[taskID](inputs)

                    # output = output1 * (mask + 1).unsqueeze(dim=1)  # Apply Mask

                    output = head[taskID](base_feat)

                    if taskID == 0:
                        _outputs_merged = output.cpu()
                    else:
                        _outputs_merged = np.hstack((_outputs_merged, output.cpu()))

                if flag_first_batch:
                    outputs = _outputs_merged
                    uuid = _uuids
                    flag_first_batch = False

                else:
                    outputs = np.vstack((outputs, _outputs_merged))
                    uuid = np.hstack((uuid, _uuids))
                pbar.update(1)
        pbar.close()
        elapsed_time = (time.time() - start_time)

        export = {}

        for i in classes['classes']:
            export[i['challenge_id']] = uuid[(-outputs)[:, i['id']].argsort()].tolist()
            # ####################################################################################### #
            # line of code above do the follow:
            # x = (-outputs)[:, i['id']         # slice the nth column from output
            # x = x.argsort()[:500]             # sort and keep 500 first ids
            # x = uuid[x].tolist()              # convert ids to uuids
            # export[i['challenge_id']] = x     # insert to export dict
            # ####################################################################################### #

        elapsedTimeString = str(round(elapsed_time, 1))

        for i in export.keys():
            export[i] = removeDuplicates(export[i])[:1000]

        _writing_file = os.path.join(_work_dir, 'export.json')
        print('Writing Export File: ', _writing_file)
        with open(_writing_file, 'w') as export_file:
            json.dump(export, export_file)

        return export, elapsedTimeString
