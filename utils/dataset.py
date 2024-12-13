import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
from PIL import Image
import torch.utils.data.dataloader as dataloader
import os
import datetime


class TrainSet(Data.Dataset):
    def __init__(self,data,label,transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        image = self.data[item]
        target = self.label[item]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, target

class CINIC10(Data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            data_dir = os.path.join(self.root, 'train')
        else:
            data_dir = os.path.join(self.root, 'test')  # or 'test' depending on your requirement

        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.train:
            image = np.array(image)
        else:
            image = self.transform(image)

        return image, label


class CustomDataset(Data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        # Print raw data shapes and types
        print(f"Original shape: {image.shape}, dtype: {image.dtype}")
        # Make sure the data type is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Printing information after data type conversion
        print(f"After conversion dtype: {image.dtype}")
        # Check and adjust the shape
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(axis=0)  # 移除单通道维度
        # Print the shape and type of processed data
        print(f"Processed shape: {image.shape}, dtype: {image.dtype}")

        # Checks if the array shape conforms to the (H, W, C) format.
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        # Print the shape and type of converted data
        print(f"Final shape before conversion: {image.shape}, dtype: {image.dtype}")
        try:
            image = Image.fromarray(image)
        except Exception as e:
            print(f"Error in converting to image: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def Dataset(cifar = False, mnist = False, fmnist = False, cinic = False, cifar100 = False, SVHN = False):
    print("data preprocessing...")
    if cifar == True:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomHorizontalFlip()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=r'../data', train=True,  transform=train_transform, download=True)

        test_set = torchvision.datasets.CIFAR10(
            root=r'../data', train=False, transform=test_transform)

        transform = train_transform
        train_img = train_set.data
        train_label = train_set.targets


    elif mnist == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(
            root=r'../data', train=True, transform=transform, download=False)

        test_set = torchvision.datasets.MNIST(
            root=r'../data', train=False, transform=transform)
        train_img = train_set.data
        train_label = train_set.targets

    elif fmnist == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.3205,))
        ])
        train_set = torchvision.datasets.FashionMNIST(
            root=r'../data', train=True, transform=transform, download=False)

        test_set = torchvision.datasets.FashionMNIST(
            root=r'../data', train=False, transform=transform)
        train_img = train_set.data
        train_label = train_set.targets
    elif cinic == True:
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
            transforms.RandomHorizontalFlip()
        ])
        # Transformer for test set
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std), ]
        )

        train_set = CINIC10(root=r'../data/CINIC-10', train=True, transform=train_transform)
        test_set = CINIC10(root=r'../data/CINIC-10', train=False, transform=test_transform)

        transform = train_transform

        train_img, train_label = np.array([s[0] for s in train_set]), np.array([int(s[1]) for s in train_set])

    elif cifar100:
        mean_cifar100 = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std_cifar100 = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_cifar100, std=std_cifar100),
            transforms.RandomHorizontalFlip()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_cifar100, std=std_cifar100),
        ])

        train_set = torchvision.datasets.CIFAR100(
            root=r'../data', train=True, transform=train_transform, download=False)

        test_set = torchvision.datasets.CIFAR100(
            root=r'../data', train=False, transform=test_transform)

        transform = train_transform
        train_img = train_set.data
        train_label = train_set.targets

    elif SVHN:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            # transforms.RandomHorizontalFlip()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = torchvision.datasets.SVHN(
            root=r'../data', split='train', download=False, transform=train_transform)

        test_set = torchvision.datasets.SVHN(
            root=r'../data', split='test', download=False, transform=test_transform)

        transform = train_transform
        train_img = train_set.data
        train_label = train_set.labels

    else:
        train_img = None
        train_label = None

        test_set = None
        transform = None

    return train_img, train_label, test_set, transform


def Data_Partition(iid, dirichlet, train_img, train_label, transform, user_num, batchSize, alpha=0.1,
                   shard=2, drop=True, classOfLabel=10):
    users_data = []
    if iid:
        udata_size = int(len(train_label) / user_num)
        for i in range(user_num):
            set = TrainSet(train_img[udata_size * i:(udata_size * (i + 1))],
                           train_label[udata_size * i:(udata_size * (i + 1))], transform)
            loader = Data.DataLoader(
                dataset=set,
                batch_size=batchSize,
                shuffle=True,
                drop_last=drop
            )
            users_data.append(loader)
    elif dirichlet:
        indexOfClients = Dirichlet(train_label, user_num, classOfLabel, alpha, min_require_size=10)
        for i in range(user_num):
            local_data = train_img[indexOfClients[i]]
            local_label = train_label[indexOfClients[i]]
            orderOfClient = np.random.permutation(local_data.shape[0])
            local_data = local_data[orderOfClient]
            local_label = local_label[orderOfClient]
            set = TrainSet(local_data, local_label, transform)
            loader = Data.DataLoader(
                dataset=set,
                batch_size=batchSize,
                shuffle=True,
                drop_last=drop
            )
            users_data.append(loader)
    else:
        # Create an array of classOfLabel to hold the index of each label's subscript.
        class_index = [[] for i in range(classOfLabel)]
        for i in range(len(train_label)):
            class_index[int(train_label[i])].append(i)
        # The classOfLabel array is further subdivided into shard arrays.
        if (shard * user_num) % classOfLabel != 0:
            print("Invalid Data Segmentation")
            interClassShardNum = 0
        else:
            interClassShardNum = int((shard * user_num) / classOfLabel)  # 表示每个类别的数据需要进一步细分为多少块
        shard_index = [[] for i in range(classOfLabel * interClassShardNum)]
        for i in range(classOfLabel):
            shard_index_temp = np.random.permutation(class_index[i])
            nnj = np.size(shard_index_temp)
            for ii in range(interClassShardNum):
                shard_index[i * interClassShardNum + ii] = shard_index_temp[int(nnj / interClassShardNum)
                                                                            * ii: int(nnj / interClassShardNum) * (
                            ii + 1)]

        order = np.arange(classOfLabel * interClassShardNum)
        np.random.shuffle(order)
        print(order)
        for i in range(user_num):
            local_data = None
            local_label = None
            for ii in range(shard):
                index_temp = shard_index[order[i * shard + ii]]
                if local_data is None:
                    local_data = train_img[index_temp]
                    local_label = train_label[index_temp]
                else:
                    local_data = np.vstack([local_data, train_img[index_temp]])
                    local_label = np.hstack([local_label, train_label[index_temp]])
            orderOfClient = np.random.permutation(local_data.shape[0])
            local_data = local_data[orderOfClient]
            local_label = local_label[orderOfClient]
            set = TrainSet(local_data, local_label, transform)
            loader = Data.DataLoader(
                dataset=set,
                batch_size=batchSize,
                shuffle=True,
                drop_last=drop
            )
            users_data.append(loader)
    print("Data processing completed.")
    return users_data


def Dirichlet(y_train, n_parties, K=10, alpha=0.1, min_require_size = 10):
    '''
    K: Number of categories
    n_parties: number of users
    min_require_size: Make sure the user has data, at least min_require_size data.
    '''
    min_size = 0
    N = y_train.shape[0]
    party2dataidx = {}
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        party2dataidx[j] = idx_batch[j]
    return party2dataidx








