import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19,resnet50,wide_resnet101_2,resnet18

nclasses = 43  # GTSRB as 43 classes

def get_resnetmodel():
    one_resnet = resnet50(pretrained=True)
    for param in one_resnet.parameters():
        param.requires_grad = True
    num_features = one_resnet.fc.in_features
    one_resnet.fc = nn.Linear(num_features, nclasses)
    return one_resnet

def get_vggmodel():
    vgg16 = vgg19(pretrained=True)
    for param in vgg16.parameters():
        param.requires_grad = True
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, nclasses)
    return vgg16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)

        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)

        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)

        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        '''
        Layer 1
        '''
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))

        '''
        Layer 2
        '''
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))

        '''
        Layer 3
        '''
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))

        '''
        Layer 4
        '''
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))

        '''
        Layer 5
        '''
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        x=F.log_softmax(x,dim=1)
        return x