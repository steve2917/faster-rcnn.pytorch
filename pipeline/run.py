import argparse
import pipeline.pipeline_methods as mth
import methods.xmlExport.xmlExport as xml

# import classifier.multi_classifier_inference as inference
# import classifier.multi_classifier_faster_inference as inference
# import classifier.attend_multi_classifier_inference as inference
import classifier.multitask_classifier_inference as inference

INPUT_FOLDER = '/home/demertzis/GitHub/tecData/data/extracted_images'
WORK_FOLDER = '/home/demertzis/GitHub/faster-rcnn.pytorch/tmp/'

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

args = parse_args()

# ###################################################################################### #
# Reads folder and creates the list with IDs
# Saves to listIDs.json
# ###################################################################################### #
ids_list = mth.readFolder(INPUT_FOLDER, _work_dir=WORK_FOLDER)


# ###################################################################################### #
# Reads fatser RCNN Inference file and creates Faster Annotation Dictionary
# Reads fasterInference.json
# Saves to faster_annotation.json
# ###################################################################################### #
# faster_anotation = mth.convertRcnnOutput(ids_list, WORK_FOLDER)


# ###################################################################################### #
# Inference model
# Exports shorted list for every class and elapsed time
# Saves to export.json
# ###################################################################################### #
export, elapsedTimeString = inference.run(ids_list, args, WORK_FOLDER)


# ###################################################################################### #
# Export to XML submission File
# ###################################################################################### #
xml.export(export, elapsedTimeString, _work_dir=WORK_FOLDER)
