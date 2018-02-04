import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self, n_filters1=32,
            n_filters2=32,
            n_fc=128,
            dropout=False):

        super(MnistModel, self).__init__()

        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        self.n_fc = n_fc
        self.dropout = dropout

        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, self.n_filters1, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(self.n_filters1, self.n_filters2, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(self.n_filters2*7*7, self.n_fc)
        self.fc2 = nn.Linear(self.n_fc, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.n_filters2*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        if self.dropout: x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)