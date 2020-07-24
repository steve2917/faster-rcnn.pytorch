import json
import torch


def class_name(_id, _classes):
    for i in _classes['classes']:
        if i['id'] == _id:
            return i['challenge_id'], i['name']


def readJSON(filename):
    with open(filename, 'r') as f:
        print('Loading: ', filename)
        data = json.loads(f.read())
        return data


def save_checkpoint(state, filename):
    torch.save(state, filename)

def display_map(ap, map, classes, experiment, total, notes, epoch, session):
    count = 0
    print('')
    print('~~~ Metrics of experiment {} - Test Set: {} Images ~~~'.format(experiment, total))
    print('~~~ Experiment: {} - Session: {} - Epoch: {} ~~~'.format(experiment, session, epoch))
    print('~~~ {} ~~~'.format(notes))
    print('~~~ mAP: {}'.format(map))
    print('id'.ljust(3), '|Class'.ljust(37), '|AP'.ljust(11))
    print('===========================================================================================')
    for i in range(len(ap)):
        class_ap = ap[i]

        _challenge_id, _class_name = class_name(count, classes)
        if _challenge_id == -1:
            _challenge_id_str = '--'
        else:
            _challenge_id_str = str(_challenge_id)

        print(_challenge_id_str.ljust(3), '|',
            _class_name.ljust(35), '|',
            "%.2f" % class_ap)
        count += 1

def display_table(cm, ap, map, classes, experiment, total, notes, epoch, session):
    count = 0
    print('')

    print('~~~ Metrics of experiment {} - Test Set: {} Images ~~~'.format(experiment, total))
    print('~~~ Experiment: {} - Session: {} - Epoch: {} ~~~'.format(experiment, session, epoch))
    print('~~~ {} ~~~'.format(notes))
    print('~~~ mAP: {}'.format(map))
    print('id'.ljust(3), '|Class'.ljust(37), '|TP'.ljust(9), '|TN'.ljust(9), '|FP'.ljust(9), '|FN'.ljust(9), '|Precision'.ljust(11), '|Recall'.ljust(11), '|AP'.ljust(11))
    print('==============================================================================================================')
    for i in range(cm.shape[0]):
        tp = cm[i][1][1]
        tn = cm[i][0][0]
        fp = cm[i][0][1]
        fn = cm[i][1][0]

        class_ap = ap[i]

        if tp != 0 or fp != 0:
            precision = tp / (tp + fp)
            precisionStr = str("%.2f" % precision)
        else:
            precision = -1
            precisionStr = '--'

        recall = tp / (tp + fn)
        _challenge_id, _class_name = class_name(count, classes)

        if _challenge_id == -1:
            _challenge_id_str = '--'
        else:
            _challenge_id_str = str(_challenge_id)

        print(_challenge_id_str.ljust(3), '|',
              _class_name.ljust(35), '|',
              str(int(tp)).ljust(7), '|',
              str(int(tn)).ljust(7), '|',
              str(int(fp)).ljust(7), '|',
              str(int(fn)).ljust(7), '|',
              precisionStr.ljust(9), '|',
              str("%.2f" % recall).ljust(9), '|',
              "%.2f" % class_ap)
        count += 1

