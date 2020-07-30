# Resnet Classifier
# Concatenate Faster RCNN results
#
#
import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes=50):
        super(ResNet, self).__init__()
        resnet_model = models.resnet50(pretrained=False, num_classes=num_classes)
        # resnet_model = models.resnet101(pretrained=False, num_classes=num_classes)
        # resnet_model = models.resnet152(pretrained=False, num_classes=num_classes)
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        # self.fc = resnet_model.fc
        # self.fc = nn.Linear(2048, num_classes)
        # for param in self.conv1.parameters():
        #      param.requires_grad = False
        #
        # for param in self.bn1.parameters():
        #     param.requires_grad = False
        #
        # for param in self.layer1.parameters():
        #     param.requires_grad = False

        # for param in self.layer2.parameters():
        #     param.requires_grad = False

        # for param in self.layer3.parameters():
        #     param.requires_grad = False

        # for param in self.layer4.parameters():
        #     param.requires_grad = False

        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x = self.fc(x)
        # result = self.sigmoid(x)

        return x


class HeadNet(nn.Module):
    def __init__(self, numClasses):
        super(HeadNet, self).__init__()
        resnet_model = models.resnet101(pretrained=False, num_classes=numClasses)

        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = nn.Linear(2048, numClasses)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x