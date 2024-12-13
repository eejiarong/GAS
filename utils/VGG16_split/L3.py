import torch
import torch.nn as nn
from torch.nn import modules



def model_selection(cifar=False, mnist=False, fmnist=False, cinic=False, cifar100=False, SVHN=False, split=False, twoLogit=False):
    num_classes = 10
    if cifar100:
        num_classes = 100
    if split:
        server_local_model = None
        if cifar or cinic or cifar100 or SVHN:
            user_model = VGG16DOWN()
            server_model = VGG16UP(num_classes=num_classes)
            if twoLogit:
                server_local_model = VGG16UP(num_classes=num_classes)
        elif mnist or fmnist:
            user_model = AlexNetDown()
            server_model = AlexNetUp()
            if twoLogit:
                server_local_model = AlexNetUp()
        else:
            user_model = None
            server_model = None
        if twoLogit:
            return user_model, server_model, server_local_model
        else:
            return user_model, server_model
    else:
        if cifar or cinic or cifar100 or SVHN:
            model = VGG16(num_classes=num_classes)
        elif mnist or fmnist:
            model = AlexNet()
        else:
            model = None

        return model


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out.view(-1, 256 * 3 * 3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out

class AlexNetDown(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class AlexNetUp(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetUp, self).__init__()
        self.model2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


class VGG16(modules.Module):

    def __init__(self, num_classes=10):
        super(VGG16,self).__init__()
        self.feature = modules.Sequential(
            # #1,
            modules.Conv2d(3,64,kernel_size=3,padding=1),
            modules.BatchNorm2d(64),
            modules.ReLU(True),
            #2
            modules.Conv2d(64,64,kernel_size=3,padding=1),
            modules.BatchNorm2d(64),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #3
            modules.Conv2d(64,128,kernel_size=3,padding=1),
            modules.BatchNorm2d(128),
            modules.ReLU(True),
            # modules.MaxPool2d(kernel_size=2,stride=2),
            #4
            modules.Conv2d(128,128,kernel_size=3,padding=1),
            modules.BatchNorm2d(128),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #5
            modules.Conv2d(128,256,kernel_size=3,padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            #6
            modules.Conv2d(256,256,kernel_size=3,padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            #7
            modules.Conv2d(256,256,kernel_size=3,padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #8
            modules.Conv2d(256,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #9
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #10
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #11
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #12
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #13
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            modules.AvgPool2d(kernel_size=1,stride=1),

            )
        self.classifier = modules.Sequential(
            # #14
            modules.Linear(512,4096),
            modules.ReLU(True),
            modules.Dropout(),
            #15
            modules.Linear(4096,4096),
            modules.ReLU(True),
            modules.Dropout(),
            #16
            modules.Linear(4096,num_classes),

        )
    def forward(self,x):
        out = self.feature(x)
        # print(out.shape),batch_size/heigth,width,
        out = out.view(out.size(0),-1)

        out = self.classifier(out)
        return out

'''4-th; 7-th; 10-th; 13-th; '''
'''The split point is the 10th convolutional layer'''

class VGG16DOWN(modules.Module):

    def __init__(self, num_classes=10):
        super(VGG16DOWN, self).__init__()
        self.feature = modules.Sequential(
            # #1,
            modules.Conv2d(3, 64, kernel_size=3, padding=1),
            modules.BatchNorm2d(64),
            modules.ReLU(True),
            # 2
            modules.Conv2d(64, 64, kernel_size=3, padding=1),
            modules.BatchNorm2d(64),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2, stride=2),
            # 3
            modules.Conv2d(64, 128, kernel_size=3, padding=1),
            modules.BatchNorm2d(128),
            modules.ReLU(True),
            # modules.MaxPool2d(kernel_size=2,stride=2),
            # 4
            modules.Conv2d(128, 128, kernel_size=3, padding=1),
            modules.BatchNorm2d(128),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2, stride=2),  # 8 × 8 × 128
            # 5
            modules.Conv2d(128, 256, kernel_size=3, padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            # 6
            modules.Conv2d(256, 256, kernel_size=3, padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            # 7
            modules.Conv2d(256, 256, kernel_size=3, padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2, stride=2),
            # 8
            modules.Conv2d(256, 512, kernel_size=3, padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            # 9
            modules.Conv2d(512, 512, kernel_size=3, padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            # 10
            modules.Conv2d(512, 512, kernel_size=3, padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x, return_features=False):
        x = self.feature(x)
        return x


class VGG16UP(modules.Module):

    def __init__(self, num_classes=10):
        super(VGG16UP, self).__init__()
        self.feature = modules.Sequential(


            # 11
            modules.Conv2d(512, 512, kernel_size=3, padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            # 12
            modules.Conv2d(512, 512, kernel_size=3, padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            # 13
            modules.Conv2d(512, 512, kernel_size=3, padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2, stride=2),
            modules.AvgPool2d(kernel_size=1, stride=1),

        )
        self.classifier = modules.Sequential(
            # #14
            modules.Linear(512, 4096),
            modules.ReLU(True),
            modules.Dropout(),
            # 15
            modules.Linear(4096, 4096),
            modules.ReLU(True),
            modules.Dropout(),
            # 16
            modules.Linear(4096, num_classes),

        )

    def forward(self, x):
        out = self.feature(x)
        # print(out.shape),batch_size/heigth,width,
        out = out.view(out.size(0), -1)

        out = self.classifier(out)
        return out

