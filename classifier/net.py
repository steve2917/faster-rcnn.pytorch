import numpy as np
import torch.nn as nn
from model.faster_rcnn.resnet import resnet

class HeadNet(nn.Module):
    def __init__(self, numClasses):
        super(HeadNet, self).__init__()
        self.objectClasses = np.asarray(['__background__', 'Pipe', 'Car',
                                     'Highway', 'Aircraft',
                                     'Building', 'Railway',
                                     'Rooftop', 'Train',
                                     'Buildings_block', 'Dam / levee',
                                     'Bridge', 'Power_line',
                                     'Boat', 'Dock',
                                     'Road', 'Container_building',
                                     'Communication_tower', 'Truck', 'Airway',
                                     'Water_tower', 'Parking_lot'])

        self.fasterRCNN = resnet(self.objectClasses, 101, pretrained=False, class_agnostic=False)
        self.fc = nn.Linear(1024, numClasses)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x