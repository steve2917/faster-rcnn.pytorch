#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json
import time
import numpy as np
from PIL import Image
#import tensorflow as tf
#import paramiko
import cv2
import os
from model.faster_rcnn.resnet import resnet
import torch
from torch.autograd import Variable
from model.utils.blob import im_list_to_blob

#from object_detection.utils import label_map_util


#Cuda visible devices
#os.environ['CUDA_VISIBLE_DEVICES']='0'

MODEL_BASE = 'models/research'

################################################################################

inputFile = 'input.json'
################################################################################

PATH_TO_CKPT = 'graph_def/faster_rcnn_resnet100_anita/exported_graphs/frozen_inference_graph.pb'
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/anita_label_map.pbtxt'

###############################################################
#class Database
class Database :
    def __init__(self, info = None, js = None):
        if js is None:
            self.info = info
            self.videos = []
            self.categories = []
        else:
            with open(js) as json_data:
                self.__dict__ = json.load(json_data)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def save2JSON(self):
        with open('results_ouput_anita_resnet.json', 'w') as outfile:
            json.dump(self,outfile, default=lambda o: o.__dict__, sort_keys=False, indent=4)
            print("json file saved successfully...")

    def PrintDatabase(self):
        print("------------------------------")
        print("info:", self.info)
        # print("images:", self.images)
        print("videos:", self.videos)
        # print("categories:", self.categories)


###############################################################
#class Info
class Info:
    def __init__(self):
        self.year = -1
        self.version =''
        self.description = ''
        self.name = ''

    def PrintInfo(self):
        print("------------------------------")
        print("year:", self.year)
        print("version:", self.version)
        print("description:", self.description)
        print("name:", self.name)

###################################################################
#class Queries
class Videos :
    def __init__(self):
        self.video_id = -1
        self.video_name = ''
        self.queries = []


    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def PrintQuery(self):
        print("------------------------------")
        print("video_id:", self.video_id)
        print("queries:", self.queries)
################################################################
#class output
class output:
    def __init__(self):
        self.info = [...]
        self.images = []
        self.annotations = []
        self.categories = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)


###################################################################
#class Queries
class Queries :
    def __init__(self):
        self.query_id = -1
        self.query_name = ''
        self.timestamp = -1
        self.annotations = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def PrintQuery(self):
        print("------------------------------")
        print("query_id:", self.query_id)
        print("query_name:", self.query_name)
        print("annotations:", self.annotations)


#####################################################################################
#class Object_Annotation
class Object_Annotation :
    def __init__(self):
        self.id = -1
        self.image_id = -1
        self.category_id = -1
        self.bbox = []
        self.score = -1.0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def PrintObject(self):
        print("------------------------------")
        print("id:", self.id)
        print("image_id:", self.image_id)
        print("category_id:", self.category_id)
        print("bbox:", self.bbox)
        print("score:", self.score)
        print("------------------------------")

#################################################################################
#class ObjectDetector
class ObjectDetector(object):

  def __init__(self):
    self.file = 'models/res101/ladi/faster_rcnn_1_120_283.pth'
    categories = np.asarray(['__background__', 'Pipe', 'Car', 'Highway', 'Aircraft', 'Building', 'Railway', 'Rooftop', 'Train', 'Buildings_block',
                    'Dam / levee', 'Bridge', 'Power_line', 'Boat', 'Dock', 'Road', 'Container_building',
                    'Communication_tower', 'Truck', 'Airway', 'Water_tower', 'Parking_lot'])
    fasterRCNN = resnet(categories, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    checkpoint = torch.load(self.file)
    fasterRCNN.load_state_dict(checkpoint['model'])

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if 1 > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        # make variable
    with torch.no_grad():
        self.im_data = Variable(im_data)
        self.im_info = Variable(im_info)
        self.num_boxes = Variable(num_boxes)
        self.gt_boxes = Variable(gt_boxes)

    fasterRCNN.cuda()

    fasterRCNN.eval()
    self.graph = fasterRCNN

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    im_in = self._load_image_into_numpy_array(image)

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
      self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      self.gt_boxes.resize_(1, 1, 5).zero_()
      self.num_boxes.resize_(1).zero_()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = self.graph(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    # image_np = self._load_image_into_numpy_array(image)
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    #
    # graph = self.detection_graph
    # image_tensor = graph.get_tensor_by_name('image_tensor:0')
    # boxes = graph.get_tensor_by_name('detection_boxes:0')
    # scores = graph.get_tensor_by_name('detection_scores:0')
    # classes = graph.get_tensor_by_name('detection_classes:0')
    # num_detections = graph.get_tensor_by_name('num_detections:0')
    #
    # (boxes, scores, classes, num_detections) = self.sess.run(
    #     [boxes, scores, classes, num_detections],
    #     feed_dict={image_tensor: image_np_expanded})
    #
    # boxes, scores, classes, num_detections = map(
    #     np.squeeze, [boxes, scores, classes, num_detections])
    num_detections= -1
    classes = -1
    return boxes, scores, classes, num_detections

# class Category
class Category:
    def __init__(self):
        self.id = -1
        self.name = ''

    def PrintCategory(self):
        print("------------------------------")
        print("id:", self.id)
        print("name:", self.name)


# #class Categories
class Categories:
    def __init__(self):
        self.categories = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def PrintCategories(self):
        print("------------------------------")
        print("categories:", self.categories)

##########################################################################
# Method detect_objects
def detect_objects(image):

    image = Image.open(image).convert('RGB')
    #image.show()
    # tf.reset_default_graph()
    boxes, scores, classes, num_detections = object_det.detect(image)
    image.thumbnail((480, 480), Image.ANTIALIAS)
    # print(scores)
    # print(classes)

    found_boxes = 0
    # get objects when prediction over 50%
    # for i, value in enumerate(scores):#:range(int(num_detections)):
    #     if scores[i] < 0.30: break
    #     found_boxes += 1
    return found_boxes, boxes, scores, classes


################################################################################
def show_stats(dataset):
    # get all images containing given categories, select one at random
    catIds = dataset.getCatIds()
    # print(catIds)

    imgIds = dataset.getImgIds()
    print('Found: '+str(len(imgIds))+' images of category: '+str(catIds[0]))
    # logging.info('Found: '+str(len(imgIds))+' images of category: '+str(catIds[0]))
    # imgIds = dataset.getImgIds(imgIds = [1000,1001])

    AnnIds = dataset.getAnnIds(catIds=catIds)
    print('Found: '+str(len(AnnIds))+' instances of category: '+str(catIds[0]))
    # logging.info('Found: '+str(len(AnnIds))+' instances of category: '+str(catIds[0]))
    print("------------------------------")

def init_Categories():
    # categories = ['Ignored_regions', 'Pedestrian', 'People', 'Bicycle', 'Car', 'Van', 'Truck', 'Tricycle',
    #               'Awning_tricycle', 'Bus', 'Motor', 'Others']
    categories = ['Pipe', 'Car', 'Highway', 'Aircraft', 'Building', 'Railway', 'Rooftop', 'Train', 'Buildings_block',
                  'Dam / levee', 'Bridge', 'Power_line', 'Boat', 'Dock', 'Road', 'Container_building',
                  'Communication_tower', 'Truck', 'Airway', 'Water_tower', 'Parking_lot']
    #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #categories = label_map_util.convert_label_map_to_categories(
    #    label_map, max_num_classes=80, use_display_name=True)
    Cats = Categories()
    for index in range(len(categories)):
        tempCat = Category()
        #tempCat.id = categories[index]['id']
        tempCat.id  = index
        tempCat.name = categories[index]
        #tempCat.name = categories[index]['name']

        Cats.categories.append(tempCat)
    return Cats
################################################################################

def get_input(input_file):
    # type: (object) -> object
    # load dataset
    # self.dataset, self.anns, self.cats, self.imgs, self.vids = dict(), dict(), dict(), dict(), dict()
    # self.imgToAnns, self.catToImgs, self.vidToAnns = defaultdict(list), defaultdict(list), defaultdict(list)
    if input_file is not None:
        print('loading input into memory...')
        tic = time.time()
        input_data = json.load(open(input_file, 'r'))
        assert type(input_data) == dict, 'input file format {} not supported'.format(type(input_data))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        return input_data


def create_index(input_file):
    imgs = {}
    if 'query' in input_file:
        for img in input_file['query']:
            imgs[img['image_id']] = img

    print('Number of queries: ' + str(len(imgs)))
    return imgs

def object2Json(found_boxes, boxes, scores, classes, img_id):
    # list with annotations
    allObjects = []
    for i in range(0,found_boxes):
        tempObject = Object_Annotation()
        tempObject.id = i+1
        tempObject.category_id = classes[i]

        bb = boxes[i].tolist()
        #bb = ["{0:.2f}".format(bb[x]) for x in range(len(bb))]
        bb = [round(bb[x],3) for x in range(len(bb))]

        tempObject.bbox = bb
        tempObject.image_id = img_id
        tempObject.score = round(scores[i],2)
        # tempObject.PrintObject()
        # tempObject.toJSON()

        allObjects.append(tempObject)
    return allObjects


def info2Json():
    # Building info
    tempInfo = Info()
    tempInfo.year =2019
    tempInfo.version = 'v0.1'
    tempInfo.description = 'Trafficking related dataset'
    tempInfo.name = 'ANITA_Dataset'
    tempInfo.PrintInfo()
    return tempInfo


def video2Json(allQueries, index, name):
    # Building Videos
    tempVideo = Videos()
    tempVideo.video_id = index + 1
    tempVideo.video_name = name
    tempVideo.queries = allQueries

    return tempVideo

def dataset2Json(allVideos, tempDataset = None):

    if tempDataset is None:
        # info
        tempInfo = info2Json()

        # Building Dataset
        tempDataset = Database(tempInfo)
        tempDataset.videos = allVideos
        tempDataset.categories = Categs.categories
    else:
        for i in range(len(allVideos)):
            tempDataset.videos.append(allVideos[i])

    # tempDataset.PrintDatabase()
    # tempDataset.queries[0].PrintQuery()
    # tempDataset.queries[1].annotations[0].PrintObject()
    # print("------------------------------")
    # print(tempDataset.toJSON())
    # print("------------------------------")
    tempDataset.save2JSON()


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
  #im_orig -= cfg.PIXEL_MEANS
  im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  #for target_size in cfg.TEST.SCALES:
  for target_size in (600,):
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    #if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    if np.round(im_scale * im_size_max) > 1000:
      #im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
      im_scale = float(1000) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

################################################################################

object_det = ObjectDetector()


if __name__ == '__main__':

    # show_stats(dataset)



    # Get input from json files
    # query_images = get_input(inputFile)
    # queries = create_index(query_images)

    # Create input list from video files
    # dir all videos to a numpy array
    # danteVideosPath = "/home/gountakos/Desktop/DANTE_DATASETS/TooManyEyes"
    # allVideoNames = [f for f in listdir(danteVideosPath) if isfile(join(danteVideosPath, f))]
    # allVideoNames = np.array(allVideoNames)

    Categs = init_Categories()
    # connect sftp
    #folderPathSource = '/home/atpsaltis/Desktop/ANITA/partners/'
    folderPathSource = 'images/'
    localPathSource = '/home/atpsaltis/Desktop/ANITA'

    # client = get_sftp()
    # sftp = client.open_sftp()
    # sftp.stat('/tmp')

    allQueries = []
    count = 0
    tempDataset = None
    # split_set = sftp.listdir(folderPathSource)
    img_set = os.listdir(folderPathSource)

    #sorted(split_set)
    tac = time.time()
    #for sp in split_set:
    #    print(sp)
        # list dir images
        # img_set = sftp.listdir(folderPathSource + sp)
    #img_set = os.listdir(folderPathSource + sp)

    sorted(img_set)

    for i, img in enumerate(img_set):

        count += 1
        #path = os.path.join(folderPathSource, sp, img)
        #local_path = os.path.join(localPathSource,img)
        # sftp.get(path, local_path)
        #keyFrame = cv2.imread(path)
        # sftp.remove(local_path)

        #img = Image.fromarray(keyFrame)
        found_boxes, boxes, scores, classes = detect_objects(os.path.join(folderPathSource, img))
        result = object2Json(found_boxes, boxes, scores, classes, count)
        tempQueries = Queries()
        tempQueries.query_id = count
        tempQueries.query_name = img
        tempQueries.timestamp = 0
        tempQueries.annotations = result
        # tempQueries.PrintQuery()
        allQueries.append(tempQueries)
        # break
    # break
    tic = time.time()

    tempVideos = video2Json(allQueries, 0, 'Anita_set')

    # save dataset into json
    if tempDataset is not None:
        dataset2Json(allVideos=tempVideos, tempDataset=tempDataset)
    else:
        dataset2Json(allVideos=tempVideos)
