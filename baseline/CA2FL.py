import copy
import sys
import os

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Add the 'utils' directory to sys.path
utils_path = os.path.join(parent_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import os
import random
import numpy as np
from network import model_selection
from dataset import Dataset, Data_Partition
import datetime

begin_time = datetime.datetime.now()



"""
CA2FL TACKLING THE DATA HETEROGENEITY IN ASYNCHRONOUS FEDERATED LEARNING WITH CACHED UPDATE CALIBRATION
"""


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Experimental parameter settings
iid = False
dirichlet = False
shard = 2
alpha = 0.1
epochs = 2000
localEpoch = 20
user_num = 20
user_parti_num = 10
batchSize = 32
lr = 0.01
# Training data selection
cifar = True
mnist = False
fmnist = False
cinic = False
cifar100 = False
SVHN = False
# Random seeds selection
seed_value = 2023
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

Accu_Test_Frequency = 1

if cifar100:
    classOfLabel = 100
else:
    classOfLabel = 10

# Simulate real communication environments
WRTT = True   # True for simulation, False for no simulation

# =========================================================
# ======   data loading and data pre-processing     =======
# =========================================================

alldata, alllabel, test_set, transform = Dataset(cifar=cifar, mnist=mnist, fmnist=fmnist, cinic=cinic,
                                                 cifar100=cifar100, SVHN=SVHN)

test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=128,
    shuffle=True
)

train_index = np.arange(0, len(alldata))
random.shuffle(train_index)
train_img = np.array(alldata)[train_index]
train_label = np.array(alllabel)[train_index]

# Partition data among users
users_data = Data_Partition(iid, dirichlet, train_img, train_label, transform,
                            user_num, batchSize, alpha, shard, drop=False, classOfLabel=classOfLabel)

# =========================================================
# ==============      initialization        ===============
# =========================================================

model = model_selection(cifar=cifar, mnist=mnist, fmnist=fmnist, cinic=cinic, cifar100=cifar100, SVHN=SVHN)
model.to(device)

userParam = copy.deepcopy(model.state_dict())

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()

# =========================================================
# ==============   Clients Definition      ================
# =========================================================
# Initialize client computing capabilities
def generate_computing(user_num):
    # 10**9 ～ 10**10 FLOPs
    return np.random.uniform(10 ** 9, 10 ** 10, user_num)


def generate_position(user_num):
    return np.random.uniform(0.1, 1, user_num)


clients_computing = generate_computing(user_num)
clients_position = generate_position(user_num)

# Calculate the communication rate for each client
w = 10 ** 7
N = 3.981 * 10 ** (-21)
rates = []
for i in range(user_num):
    path_loss = 128.1 + 37.6 * np.log10(clients_position[i])
    h = 10 ** (-path_loss / 10)
    rates.append(w * np.log2(1 + (0.2 * h / (w * N))))

# clients_computing = [3897894735.92771, 9013802066.105328, 6292470299.054264, 2139364841.5386212, 2272071003.1736717,
#                      5211060330.473793, 1198806954.5634105, 7545472413.904353, 5719486080.450004, 5904417149.294411,
#                      5107359343.831957, 5512440381.97626, 4550216975.213522, 2360550729.2126856, 4247876594.1973186,
#                      2458693061.80659, 4041628237.5291243, 2622909526.838994, 4518922610.259291, 1320833893.6367936]
# clients_position = [0.6083754840266261, 0.2831153426857548, 0.38854401189901344, 0.4389074044864103, 0.2656487276448098,
#                     0.19355665271763545, 0.509434502092531, 0.2762774537423048, 0.4406728802978894, 0.9374787633533964,
#                     0.7841437418818485, 0.7936878138929144, 0.6370305015692377, 0.8124590324955729, 0.8293044732903868,
#                     0.9825015031837409, 0.8963067286477621, 0.19882101292753768, 0.8377396846064555, 0.3768516013443233]
# rates = [25948858.23931885, 64992952.13823921, 48181797.74664864, 41864543.70309627, 68413272.50342345,
#          85499773.20790398, 34364550.5456598, 66305169.93391083, 41658828.81171437, 9941368.164306499,
#          15564920.989243941, 15135144.574422104, 23896461.265810642, 14324059.624996226, 13633385.95929576,
#          8730296.888630323, 11205899.934015611, 84048199.78995569, 13300281.995040257, 49783651.93687485]

if WRTT is True:
    print(clients_computing)
    print(clients_position)
    print(rates)

class Client:
    def __init__(self, user_data, local_epoch, gradient, minibatch=0, computing=0, rate=0, time=0):
        self.user_data = user_data
        self.dataloader_iter = iter(user_data)
        self.local_epoch = local_epoch
        self.count = 0  # Record the number of local iterations
        self.iter_num = 0  # Record the number of times the user's data has been iterated
        self.gradient = gradient

        self.minibatch = minibatch
        self.computing = computing
        self.rate = rate
        self.time = time

    def increment_counter(self):
        # Record the number of local iterations
        self.count += 1
        if self.count == self.local_epoch:
            self.count = 0
            return True
        return False

    def train_one_iteration(self):
        try:
            data = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.user_data)
            data = next(self.dataloader_iter)
        return data

    def model_process(self):
        workload = 69833728  # Workload of one image FLOPs
        self.time += (workload / self.computing)

    def transmit_model(self):
        model_volume = 123799872
        self.time += (model_volume / self.rate)


gradient = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
if WRTT is True:
    clients = [Client(users_data[i], localEpoch, copy.deepcopy(gradient), batchSize, clients_computing[i], rates[i], 0) for i in range(user_num)]
else:
    clients = [Client(users_data[i], localEpoch, copy.deepcopy(gradient)) for i in range(user_num)]

total_data_points = sum([len(users_data[r].dataset) for r in range(user_num)])
fed_avg_freqs = [len(users_data[r].dataset) / total_data_points for r in range(user_num)]
print(fed_avg_freqs)

# =========================================================
# ================       training        ==================
# =========================================================

def replace_user(order, k, user_num):
    available_users = set(range(user_num)) - set(order)
    new_user = np.random.choice(list(available_users))
    order[np.where(order == k)[0][0]] = new_user
    return order

def find_client_with_min_time(clients, order):
    min_time = float('inf')
    client_with_min_time = None
    for index in order:
        client = clients[index]
        if client.time < min_time:
            min_time = client.time
            client_with_min_time = index
    return client_with_min_time


total_accuracy = []
total_loss = []
time_record = []
current_time = 0

epoch = 0
order = np.random.choice(range(user_num), user_parti_num, replace=False)
if WRTT is True:   # Initialize the training time
    for i in order:
        clients[i].time = current_time
        for ee in range(localEpoch):
            clients[i].model_process()
            clients[i].model_process()
        clients[i].transmit_model()

usersParam = []
for i in range(user_parti_num):
    usersParam.append(copy.deepcopy(userParam))
count_local = 0

while epoch != epochs:
    model.train()

    '''Select a client to train'''
    if WRTT is True:
        selected_client = find_client_with_min_time(clients, order)
        current_time = clients[selected_client].time
    else:
        selected_client = np.random.choice(order)

    model.load_state_dict(usersParam[np.where(order == selected_client)[0][0]], strict=True)
    for iteration in range(localEpoch):
        images, labels = clients[selected_client].train_one_iteration()
        images = images.to(device)
        labels = labels.to(device)
        final_output = model(images)
        loss = criterion(final_output, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
        optimizer.step()

    '''model aggregation'''
    count_local += 1
    for key in usersParam[np.where(order == selected_client)[0][0]]:
        clients[selected_client].gradient[key] = model.state_dict()[key].detach().clone() - \
                                                 usersParam[np.where(order == selected_client)[0][0]][key].detach().clone()

    # Update the client model if the buffer is full
    if count_local == user_parti_num:
        count_local = 0

        for i in range(user_num):
            for key in userParam:
                userParam[key] = userParam[key] + clients[i].gradient[key].detach().clone() * fed_avg_freqs[i]

        test_flag = ((epoch + 1) % Accu_Test_Frequency == 0)
        if WRTT is True and test_flag:
            time_record.append(current_time)
            print("Time: "+str(current_time))

        # =========================================================
        # ============       testing per round        =============
        # =========================================================
        epoch += 1
        if test_flag:
            model.eval()
            model.load_state_dict(userParam, strict=True)
            with torch.no_grad():
                correct = 0
                total = 0
                for (images, labels) in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuary = correct / total
            total_accuracy.append(accuary)
            print("Global iteration: " + str(epoch))
            print("Accuracy: ",  total_accuracy[-1])
            print()

    index = np.where(order == selected_client)[0][0]
    usersParam[index] = userParam
    order = replace_user(order, selected_client, user_num)
    if WRTT is True:
        clients[order[index]].time = current_time
        for ee in range(localEpoch):
            clients[order[index]].model_process()
            clients[order[index]].model_process()
        clients[order[index]].transmit_model()

print(time_record)
print(total_accuracy)
time_record_str = ', '.join(str(x) for x in time_record)
total_accuracy_str = ', '.join(str(x) for x in total_accuracy)
print('time = [' + time_record_str + ']')
print('CA2FL = [' + total_accuracy_str + ']')
print(total_loss)

# 获取当前时间
end_time = datetime.datetime.now()

# 将时间转换为字符串
begin_time_str = begin_time.strftime("%Y-%m-%d %H:%M:%S")
end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

selectDataset = "cifar10" if cifar else "mnist" if mnist else "fmnist" if fmnist else "cinic" if cinic else "cifar100" \
    if cifar100 else "SVHN" if SVHN else "None"


with open('CA2FL.txt', 'w') as f:
    # 写入字符串
    if dirichlet:
        f.write(
            f'seed_value: {seed_value}; alpha: {alpha}; epochs: {epochs}; {selectDataset}; local epoch: {localEpoch}; \n'
            f'num of clients: {user_num}; num of participating clients: {user_parti_num}; batchsize: {batchSize}; learning rate: {lr}; \n')
    else:
        f.write(
            f'seed_value: {seed_value}; shard: {shard}; epochs: {epochs}; {selectDataset}; local epoch: {localEpoch}; \n'
            f'num of clients: {user_num}; num of participating clients: {user_parti_num}; batchsize: {batchSize}; learning rate: {lr}; \n')
    f.write(begin_time_str + ' ~ ' + end_time_str + '\n')
    if WRTT is True:
        clients_computing_str = ', '.join(str(x) for x in clients_computing)
        clients_position_str = ', '.join(str(x) for x in clients_position)
        rates_str = ', '.join(str(x) for x in rates)
        f.write('clients computing = [' + clients_computing_str + ']\n')
        f.write('clients position = [' + clients_position_str + ']\n')
        f.write('clients rates = [' + rates_str + ']\n')
        f.write('time = [' + time_record_str + ']\n')
    f.write('CA2FL = [' + total_accuracy_str + ']\n')

