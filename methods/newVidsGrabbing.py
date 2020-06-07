import cv2
import os
import json


# def readJSON(filename):
#     with open(filename, 'r') as f:
#         triplets = json.loads(f.read())
#     return triplets
# Read the video from specified path


def exportFrames(_video, _frame_space, _writeDirectory, _filename):
    _currentframe = 0
    _count = 0

    while True:
        # reading from frame
        ret, frame = video.read()

        if ret:
            # if video is still left continue creating images
            if _currentframe % frame_space == 0:
                _count += 1
                #name = video_file_full_noExt + "/frame" + str(_count).zfill(7) + '.jpg'
                name = _writeDirectory + "images/" + _filename + '_' + str(_count).zfill(2) + '.jpg'
                cv2.imwrite(name, frame)
                print('Creating...' + name, ret)

            # writing the extracted images

            # increasing counter so that it will
            # show how many frames are created
            _currentframe += 1
        else:
            return _count

def getVideoList(_readDirectory, _ext):
    _files = []
    for _file in os.listdir(readDirectory):
        _split = _file.split('.')
        if len(_split) > 1 and _split[1] == _ext:
            _files.append(_file)
    return _files

readDirectory = "/home/demertzis/tmp/"

frame_space = 300
vids = dict()

files = getVideoList(readDirectory, 'mp4')

for i in range(len(files)):
    video_file_full = readDirectory + files[i]  # '/home/demertzis/tmp/folderA/Shot8_015.mp4'
    video_file_full_noExt = video_file_full.split('.')[0]  # '/home/demertzis/tmp/folderA/Shot8_015'

    video = cv2.VideoCapture(video_file_full)
    # try:
    #     if not os.path.exists(video_file_full_noExt):
    #         os.makedirs(video_file_full_noExt)
    # except OSError:
    #     print('Error: Creating directory of data')

    count = exportFrames(video, frame_space, readDirectory, files[i].split('.')[0])

    vids[files[i]] = count
    # Release all space and windows once done
    video.release()

with open('newVids.json', 'w') as fp:
    json.dump(vids, fp, indent=2)
