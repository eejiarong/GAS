import torch
import torch.nn as nn
import numpy as np


def model_selection(cifar=False, mnist=False, fmnist=False, cinic=False, cifar100=False, SVHN=False, split=False, twoLogit=False):
    num_classes = 10
    if cifar100:
        num_classes = 100
    if split:
        server_local_model = None
        if cifar or cinic or cifar100 or SVHN:
            user_model = AlexNetDownCifar()
            server_model = AlexNetUpCifar(num_classes=num_classes)
            if twoLogit:
                server_local_model = AlexNetUpCifar(num_classes=num_classes)
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
            model = AlexNetCifar(num_classes=num_classes)
        elif mnist or fmnist:
            model = AlexNet()
        else:
            model = None

        return model

def inversion_model(feature_size=None):
    model = custom_AE(input_nc=64, output_nc=3, input_dim=8, output_dim=32)
    return model


class custom_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            #TODO: change to Conv2d
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class AlexNetCifar(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetCifar, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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
            nn.Linear(512, num_classes),
        )

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out.view(-1, 256*3*3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out

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
        features = out.view(-1, 256*3*3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out

class AlexNetDownCifar(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDownCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class AlexNetUpCifar(nn.Module):
    def __init__(self, width_mult=1, num_classes=10):
        super(AlexNetUpCifar, self).__init__()
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
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.model2(x)
        x = x.view(-1, 256*3*3)
        x = self.classifier(x)
        return x

class AlexNetDown(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
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
        x = x.view(-1, 256*3*3)
        x = self.classifier(x)
        return x
