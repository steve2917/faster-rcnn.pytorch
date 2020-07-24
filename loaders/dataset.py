import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class LadiDataset(Dataset):
    # Characterizes a dataset for PyTorch

    def __init__(self, list_IDs, labels, transforms=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms = transforms

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        uuid = self.list_IDs[index]

        X = Image.open('../data/images/' + uuid + '.jpg')
        X = self.transforms(X)
        X = np.array(X)
        X = torch.from_numpy(X)

        y = self.labels[uuid]
        y = np.array(y, dtype=np.float32)
        y = torch.from_numpy(y)

        return X, y

class LadiDatasetMultiInput(Dataset):
    # Characterizes a dataset for PyTorch

    def __init__(self, list_IDs, labels, faster=None, features=None, transforms=None):
        # ' Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms = transforms
        self.faster = faster
        self.features = features

    def __len__(self):
        # ' Denotes the total number of samples '
        return len(self.list_IDs[0])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = []
        y = []
        f_infrastructure, f_vehicle = [], []
        for _taskID in range(len(self.list_IDs)):
            uuid = self.list_IDs[_taskID][index]
            im_file = './data/ladi/images/train/' + uuid + '.jpg'
            # _X = Image.open('./data/ladi/images/train/' + uuid + '.jpg')
            # _X = self.transforms(_X)
            # _X = np.array(_X)
            # _X = torch.from_numpy(_X)
            # X.append(_X)

            X.append(im_file)


            if self.faster:
                _f = self.faster[uuid]
                f = []
                for _feature in self.features:
                    f.append(_f[_feature])

                f = np.array(f, dtype=np.float32)
                f_infrastructure = f[:9]
                f_vehicle = f[9:]
                f_infrastructure = torch.from_numpy(f_infrastructure)
                f_vehicle = torch.from_numpy(f_vehicle)

            _y = self.labels[_taskID][uuid]
            _y = np.array(_y, dtype=np.float32)
            _y = torch.from_numpy(_y)
            y.append(_y)

        return X, y, f_infrastructure, f_vehicle

class LadiDatasetWithFasterOutput(Dataset):
    # Characterizes a dataset for PyTorch

    def __init__(self, list_IDs, labels, faster, transforms=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms = transforms
        self.faster = faster

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        uuid = self.list_IDs[index]

        X = Image.open('../data/images/' + uuid + '.jpg')
        X = self.transforms(X)
        X = np.array(X)
        X = torch.from_numpy(X)

        m = self.faster[uuid]
        m = np.array(m, dtype=np.float32)
        m = torch.from_numpy(m)

        y = self.labels[uuid]
        y = np.array(y, dtype=np.float32)
        y = torch.from_numpy(y)

        return X, m, y

class LadiDatasetInference(Dataset):
    def __init__(self, list_IDs, faster=None, features=None, transforms=None):
        self.list_IDs = list_IDs
        self.transforms = transforms
        self.faster = faster
        self.features = features

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        uuid = self.list_IDs[index]
        f_infrastructure, f_vehicle = [], []

        # X = Image.open('../data/images/' + uuid + '.jpg')
        X = Image.open('../data/extracted_images/' + uuid + '.jpg')
        X = self.transforms(X)
        X = np.array(X)
        X = torch.from_numpy(X)

        if self.faster:
            _f = self.faster[uuid]
            f = []
            for _feature in self.features:
                f.append(_f[_feature])

            f = np.array(f, dtype=np.float32)
            f_infrastructure = f[:9]
            f_vehicle = f[9:]
            f_infrastructure = torch.from_numpy(f_infrastructure)
            f_vehicle = torch.from_numpy(f_vehicle)

        return uuid, X, f_infrastructure, f_vehicle
