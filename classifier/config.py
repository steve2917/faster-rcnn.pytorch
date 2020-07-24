from easydict import EasyDict as edict
import os
import argparse

__C = edict()
__C.CREATE_ANNOTATION_FILE = edict()
__C.SPLITTER = edict()
__C.TRAIN_CLASSIFIER = edict()
__C.MULTI_TASK_TRAIN = edict()
__C.MULTI_TASK_TEST = edict()
__C.MULTI_CLASSIFIER_TEST = edict()
__C.MULTI_CLASSIFIER_TRAIN = edict()
__C.MULTI_CLASSIFIER_INFERENCE = edict()
__C.MULTI_CLASSIFIER_FASTER_TEST = edict()

# Consumers can get config by:
# from fast_rcnn_config import cfg
create_annotation_file_cfg = __C.CREATE_ANNOTATION_FILE
splitter_cfg = __C.SPLITTER
train_classifier_cfg = __C.TRAIN_CLASSIFIER
multi_task_train_cfg = __C.MULTI_TASK_TRAIN
multi_task_test_cfg = __C.MULTI_TASK_TEST
multi_classifier_test_cfg = __C.MULTI_CLASSIFIER_TEST
multi_classifier_train_cfg = __C.MULTI_CLASSIFIER_TRAIN
multi_classifier_inference_cfg = __C.MULTI_CLASSIFIER_INFERENCE
multi_classifier_faster_test_cfg = __C.MULTI_CLASSIFIER_FASTER_TEST

# ##################################################################################### #
# General Configuration
# ##################################################################################### #
__C.TASKS_FOLDER = 'tasks/'
__C.ANNOTATION_FOLDER = 'data/annotation/'
__C.MODEL_FOLDER = 'models/'


# ##################################################################################### #
# create_annotation_file settings
# CREATE_ANNOTATION_FILE
# ##################################################################################### #
__C.CREATE_ANNOTATION_FILE.CATEGORY = 'damage'
__C.CREATE_ANNOTATION_FILE.CLASS_NUM = 7

# Folders and Files
__C.CREATE_ANNOTATION_FILE.PATH_TO_ROOT = '../'
__C.CREATE_ANNOTATION_FILE.TASKS_FOLDER = os.path.join(__C.CREATE_ANNOTATION_FILE.PATH_TO_ROOT, __C.TASKS_FOLDER)

__C.CREATE_ANNOTATION_FILE.CLASSES_FILE = os.path.join(
    __C.CREATE_ANNOTATION_FILE.TASKS_FOLDER,
    __C.CREATE_ANNOTATION_FILE.CATEGORY,
    "classes_{}.json".format(__C.CREATE_ANNOTATION_FILE.CATEGORY)
)

__C.CREATE_ANNOTATION_FILE.ANNOTATION_FOLDER = os.path.join(__C.CREATE_ANNOTATION_FILE.PATH_TO_ROOT, __C.ANNOTATION_FOLDER)

__C.CREATE_ANNOTATION_FILE.METADATA_FILE = os.path.join(
    __C.CREATE_ANNOTATION_FILE.ANNOTATION_FOLDER,
    'ladi_images_metadata.csv'
)

__C.CREATE_ANNOTATION_FILE.RESPONSES_FILE = os.path.join(
    __C.CREATE_ANNOTATION_FILE.ANNOTATION_FOLDER,
    'ladi_aggregated_responses.tsv'
)

__C.CREATE_ANNOTATION_FILE.FASTER_INFERENCE_FILE = os.path.join(
    __C.CREATE_ANNOTATION_FILE.ANNOTATION_FOLDER,
    'results_output_ladi_classes.json'
)

# ##################################################################################### #
# SPLITTER Settings
# ##################################################################################### #
__C.SPLITTER.SEED = 4
__C.SPLITTER.SPLIT_PARAMETER = 0.9
__C.SPLITTER.CATEGORY = 'damage'

# Folders and Files
__C.SPLITTER.PATH_TO_ROOT = '../'
__C.SPLITTER.TASKS_FOLDER = os.path.join(__C.SPLITTER.PATH_TO_ROOT, __C.TASKS_FOLDER)

__C.SPLITTER.LIST_IDS_FILE = os.path.join(
    __C.SPLITTER.TASKS_FOLDER,
    __C.SPLITTER.CATEGORY,
    "listIDs_{}.json".format(__C.SPLITTER.CATEGORY)
)

# ##################################################################################### #
# TRAIN_CLASSIFIER Settings
# ##################################################################################### #
def filePath(_prefix):
    return os.path.join(
        __C.TRAIN_CLASSIFIER.TASKS_FOLDER,
        __C.TRAIN_CLASSIFIER.CATEGORY,
        '{}_{}.json'.format(_prefix, __C.TRAIN_CLASSIFIER.CATEGORY))

__C.TRAIN_CLASSIFIER.CATEGORY = 'infrastructure'

# Folders and Files
__C.TRAIN_CLASSIFIER.PATH_TO_ROOT = '../'
__C.TRAIN_CLASSIFIER.TASKS_FOLDER = os.path.join(__C.TRAIN_CLASSIFIER.PATH_TO_ROOT, __C.TASKS_FOLDER)

__C.TRAIN_CLASSIFIER.LABELS_FILE = filePath("labels")
__C.TRAIN_CLASSIFIER.CLASSES_FILE = filePath("classes")
__C.TRAIN_CLASSIFIER.FASTER_ANNOTATION_FILE = filePath("faster_annotation")
__C.TRAIN_CLASSIFIER.LIST_IDS_TRAIN_FILE = filePath("listIDs_train")
__C.TRAIN_CLASSIFIER.LIST_IDS_TEST_FILE = filePath("listIDs_test")


# ##################################################################################### #
# MULTI_TASK_TRAIN Settings
# ##################################################################################### #
def filePath(_prefix):
    return os.path.join(
        __C.MULTI_TASK_TRAIN.TASKS_FOLDER,
        __C.MULTI_TASK_TRAIN.CATEGORY,
        '{}_{}.json'.format(_prefix, __C.TRAIN_CLASSIFIER.CATEGORY))


__C.MULTI_TASK_TRAIN.CATEGORY = 'aggregate'

# Folders and Files
__C.MULTI_TASK_TRAIN.PATH_TO_ROOT = './'
__C.MULTI_TASK_TRAIN.TASKS_FOLDER = os.path.join(__C.MULTI_TASK_TRAIN.PATH_TO_ROOT, __C.TASKS_FOLDER)

__C.MULTI_TASK_TRAIN.MODEL_FOLDER = os.path.join(
    __C.MULTI_TASK_TRAIN.PATH_TO_ROOT,
    __C.MODEL_FOLDER
)

__C.MULTI_TASK_TRAIN.LABELS_FILE = filePath("labels")
__C.MULTI_TASK_TRAIN.CLASSES_FILE = filePath("classes")
__C.MULTI_TASK_TRAIN.FASTER_ANNOTATION_FILE = filePath("faster_annotation")
__C.MULTI_TASK_TRAIN.LIST_IDS_TRAIN_FILE = filePath("listIDs_train")
__C.MULTI_TASK_TRAIN.LIST_IDS_TEST_FILE = filePath("listIDs_test")


# ##################################################################################### #
# MULTI_TASK_TEST Settings
# ##################################################################################### #
def filePath(_prefix):
    return os.path.join(
        __C.MULTI_TASK_TEST.TASKS_FOLDER,
        __C.MULTI_TASK_TEST.CATEGORY,
        '{}_{}.json'.format(_prefix, __C.MULTI_TASK_TEST.CATEGORY))

__C.MULTI_TASK_TEST.CATEGORY = 'aggregate'

# Folders and Files
__C.MULTI_TASK_TEST.PATH_TO_ROOT = '../'
__C.MULTI_TASK_TEST.TASKS_FOLDER = os.path.join(__C.MULTI_TASK_TEST.PATH_TO_ROOT, __C.TASKS_FOLDER)

__C.MULTI_TASK_TEST.MODEL_FOLDER = os.path.join(
    __C.MULTI_TASK_TEST.PATH_TO_ROOT,
    __C.MODEL_FOLDER
)

__C.MULTI_TASK_TEST.LABELS_FILE = filePath("labels")
__C.MULTI_TASK_TEST.CLASSES_FILE = filePath("classes")
__C.MULTI_TASK_TEST.FASTER_ANNOTATION_FILE = filePath("faster_annotation")
__C.MULTI_TASK_TEST.LIST_IDS_TRAIN_FILE = filePath("listIDs_train")
__C.MULTI_TASK_TEST.LIST_IDS_TEST_FILE = filePath("listIDs_test")


# ##################################################################################### #
# MULTI_CLASSIFIER_TEST Settings
# ##################################################################################### #
def filePath(_prefix):
    return os.path.join(
        __C.MULTI_CLASSIFIER_TEST.TASKS_FOLDER,
        __C.MULTI_CLASSIFIER_TEST.CATEGORY,
        '{}_{}.json'.format(_prefix, __C.MULTI_CLASSIFIER_TEST.CATEGORY))


__C.MULTI_CLASSIFIER_TEST.CATEGORY = 'aggregate'

# Folders and Files
__C.MULTI_CLASSIFIER_TEST.PATH_TO_ROOT = '../'
__C.MULTI_CLASSIFIER_TEST.TASKS_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_TEST.PATH_TO_ROOT,
    __C.TASKS_FOLDER
)

__C.MULTI_CLASSIFIER_TEST.MODEL_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_TEST.PATH_TO_ROOT,
    __C.MODEL_FOLDER
)

__C.MULTI_CLASSIFIER_TEST.LABELS_FILE = filePath("labels")
__C.MULTI_CLASSIFIER_TEST.CLASSES_FILE = filePath("classes")
__C.MULTI_CLASSIFIER_TEST.FASTER_ANNOTATION_FILE = filePath("faster_annotation")
__C.MULTI_CLASSIFIER_TEST.LIST_IDS_TRAIN_FILE = filePath("listIDs_train")
__C.MULTI_CLASSIFIER_TEST.LIST_IDS_TEST_FILE = filePath("listIDs_test")


# ##################################################################################### #
# MULTI_CLASSIFIER_TRAIN Settings
# ##################################################################################### #
def filePath(_prefix):
    return os.path.join(
        __C.MULTI_CLASSIFIER_TRAIN.TASKS_FOLDER,
        __C.MULTI_CLASSIFIER_TRAIN.CATEGORY,
        '{}_{}.json'.format(_prefix, __C.MULTI_CLASSIFIER_TRAIN.CATEGORY))

__C.MULTI_CLASSIFIER_TRAIN.CATEGORY = 'infrastructure'

# Folders and Files
__C.MULTI_CLASSIFIER_TRAIN.PATH_TO_ROOT = '../'
__C.MULTI_CLASSIFIER_TRAIN.TASKS_FOLDER = os.path.join(__C.MULTI_CLASSIFIER_TRAIN.PATH_TO_ROOT, __C.TASKS_FOLDER)

__C.MULTI_CLASSIFIER_TRAIN.MODEL_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_TRAIN.PATH_TO_ROOT,
    __C.MODEL_FOLDER
)

__C.MULTI_CLASSIFIER_TRAIN.LABELS_FILE = filePath("labels")
__C.MULTI_CLASSIFIER_TRAIN.CLASSES_FILE = filePath("classes")
__C.MULTI_CLASSIFIER_TRAIN.FASTER_ANNOTATION_FILE = filePath("faster_annotation")
__C.MULTI_CLASSIFIER_TRAIN.LIST_IDS_TRAIN_FILE = filePath("listIDs_train")
__C.MULTI_CLASSIFIER_TRAIN.LIST_IDS_TEST_FILE = filePath("listIDs_test")


# ##################################################################################### #
# MULTI_CLASSIFIER_INFERENCE Settings
# ##################################################################################### #

# Folders and Files
__C.MULTI_CLASSIFIER_INFERENCE.PATH_TO_ROOT = '../'
__C.MULTI_CLASSIFIER_INFERENCE.TASKS_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_INFERENCE.PATH_TO_ROOT,
    __C.TASKS_FOLDER
)

__C.MULTI_CLASSIFIER_INFERENCE.MODEL_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_INFERENCE.PATH_TO_ROOT,
    __C.MODEL_FOLDER
)

__C.MULTI_CLASSIFIER_INFERENCE.LIST_IDS_INFERENCE_FILE = os.path.join(
    __C.MULTI_CLASSIFIER_INFERENCE.PATH_TO_ROOT,
    # 'output/listIDs_inference.json'
    'output/listIDs.json'
)


# ##################################################################################### #
# MULTI_CLASSIFIER_FASTER_TEST Settings
# ##################################################################################### #
def filePath(_prefix):
    return os.path.join(
        __C.MULTI_CLASSIFIER_FASTER_TEST.TASKS_FOLDER,
        __C.MULTI_CLASSIFIER_FASTER_TEST.CATEGORY,
        '{}_{}.json'.format(_prefix, __C.MULTI_CLASSIFIER_FASTER_TEST.CATEGORY))


__C.MULTI_CLASSIFIER_FASTER_TEST.CATEGORY = 'aggregate'

# Folders and Files
__C.MULTI_CLASSIFIER_FASTER_TEST.PATH_TO_ROOT = '../'
__C.MULTI_CLASSIFIER_FASTER_TEST.TASKS_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_FASTER_TEST.PATH_TO_ROOT,
    __C.TASKS_FOLDER
)

__C.MULTI_CLASSIFIER_FASTER_TEST.MODEL_FOLDER = os.path.join(
    __C.MULTI_CLASSIFIER_FASTER_TEST.PATH_TO_ROOT,
    __C.MODEL_FOLDER
)

__C.MULTI_CLASSIFIER_FASTER_TEST.LABELS_FILE = filePath("labels")
__C.MULTI_CLASSIFIER_FASTER_TEST.CLASSES_FILE = filePath("classes")
__C.MULTI_CLASSIFIER_TRAIN.FASTER_ANNOTATION_FILE = filePath("faster_annotation")
# __C.MULTI_CLASSIFIER_FASTER_TEST.FASTER_ANNOTATION_FILE = os.path.join(__C.ANNOTATION_FOLDER, 'results_output_ladi_classes.json')
__C.MULTI_CLASSIFIER_FASTER_TEST.LIST_IDS_TRAIN_FILE = filePath("listIDs_train")
__C.MULTI_CLASSIFIER_FASTER_TEST.LIST_IDS_TEST_FILE = filePath("listIDs_test")
