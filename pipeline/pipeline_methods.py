import os
import json


def readJSON(filename):
    with open(filename, 'r') as f:
        # print('Loading: ', filename)
        data = json.loads(f.read())
        return data


def readFolder(_readDirectory, _ext='jpg', _work_dir='/home/demertzis/GitHub/tecData/tmp'):
    print('Reading Folder: ', _readDirectory)
    _files = []

    for _file in os.listdir(_readDirectory):
        _split = _file.split('.')
        if len(_split) > 1 and _split[1] == _ext:
            _files.append(_split[0])
            a=2

    _writing_file = os.path.join(_work_dir, 'listIDs.json')

    print('Writing IDs File: ', _work_dir, _writing_file)
    with open(_writing_file, 'w') as list_file:
        json.dump(_files, list_file)

    return _files


def convertRcnnOutput(_idsList, _work_dir='/home/demertzis/GitHub/tecData/tmp'):
    _fasterInferenceFile = os.path.join(_work_dir, 'fasterInference.json')
    print('Loading Faster Inference File: ', _fasterInferenceFile)
    faster_inference = readJSON(_fasterInferenceFile)['images']
    faster_anotation = {}

    h = len(_idsList)
    for i in range(h):
        _uuid = _idsList[i]
        faster_anotation[_uuid] = next(
            item['cls_prob'] for item in faster_inference if item['file_name'] == '{}.jpg'.format(_uuid)
        )

    _writing_file = os.path.join(_work_dir, 'faster_annotation.json')
    print('Writing Faster Annotation File: ',_writing_file)
    with open(_writing_file, 'w') as faster_export_file:
        json.dump(faster_anotation, faster_export_file)

    return faster_anotation

