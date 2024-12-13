import torch.nn as nn


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
        features = out.view(-1, 256 * 3 * 3)
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

'''Split points are selected at the third convolutional layer'''


class AlexNetDownCifar(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDownCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),


        )

    def forward(self, x):
        x = self.model(x)
        return x


class AlexNetUpCifar(nn.Module):
    def __init__(self, width_mult=1, num_classes=10):
        super(AlexNetUpCifar, self).__init__()
        self.model2 = nn.Sequential(
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
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x



