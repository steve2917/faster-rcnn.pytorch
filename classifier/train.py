import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


def check_cuda():
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    return is_cuda


path = '/home/demertzis/datasets/ladi/images/'

# create dataloaders
tfms = transforms.Compose([transforms.Resize((512, 512)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.458, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])

train = ImageFolder('images/', tfms)
valid = ImageFolder('images/', tfms)

train_data_loader = DataLoader(train, batch_size=64, num_workers=4)
valid_data_loader = DataLoader(valid, batch_size=64, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 8, 3),  # inp (3, 512, 512)
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU()  # op (16, 256, 256)
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),  # inp (16, 256, 256)
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(4),
            nn.ReLU()  # op (32, 64, 64)
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),  # inp (32, 64, 64)
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU()  # op (64, 32, 32)
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),  # inp (64, 32, 32)
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU()  # op (128, 16, 16)
        )
        self.Lin1 = nn.Linear(15488, 15)
        self.Lin2 = nn.Linear(1500, 150)
        self.Lin3 = nn.Linear(150, 15)

    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = x.view(x.size(0), -1)
        x = self.Lin1(x)

        return F.log_softmax(x, dim=1)


model = Net()
# model = torch.load('800ephsapparel')
# model = model.train()
is_cuda = check_cuda()
if is_cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
for epoch in tqdm(range(50)):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    # if i % 10 == 0:    # print every 10 mini-batches
    #    print('[%d, %5d] loss: %.3f' %
    #         (epoch + 1, i + 1, running_loss / 2000))
    #  running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in valid_data_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

# torch.save(model, '850ephsapparel')