import cv2
import os
import json

from skimage import io
from skimage.transform import resize
import tarfile

# def readJSON(filename):
#     with open(filename, 'r') as f:
#         triplets = json.loads(f.read())
#     return triplets
# Read the video from specified path

def size(x, y, new):
    if x > y:
        dv = x / new
        y = y // dv
        return (new, int(y))
    else:
        dv = y / new
        x = x // dv
        return (int(x), new)


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


                try:
                    frame_resized = cv2.resize(frame, size(frame.shape[1], frame.shape[0], 1024))
                    cv2.imwrite(name, frame_resized)
                    print('Creating...' + name, ret)
                except:
                    print('Resizing error')

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

writeDirectory = "/home/demertzis/tmp/"
readDirectory = "/home/demertzis/datasets/ladi/videos/dsdi.2020.testing.shots/"

frame_space = 3
vids = dict()

files = getVideoList(readDirectory, 'mp4')

for i in range(len(files)):
    video_file_full = readDirectory + files[i]  # '/home/demertzis/tmp/folderA/Shot8_015.mp4'
    video_file_full_noExt = video_file_full.split('.')[0]  # '/home/demertzis/tmp/folderA/Shot8_015'

    video = cv2.VideoCapture(video_file_full)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(files[i], "The frame rate is: ", str(fps))

    if fps < 10:
        frame_space = 3
    elif fps < 20:
        frame_space = 100
    else:
        frame_space = 200
    # try:
    #     if not os.path.exists(video_file_full_noExt):
    #         os.makedirs(video_file_full_noExt)
    # except OSError:
    #     print('Error: Creating directory of data')

    count = exportFrames(video, frame_space, writeDirectory, files[i].split('.')[0])

    vids[files[i]] = count
    # Release all space and windows once done
    video.release()

with open('newVids.json', 'w') as fp:
    json.dump(vids, fp, indent=2)
