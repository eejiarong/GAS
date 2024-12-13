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
import network
from dataset import Dataset, Data_Partition
import datetime
from torch.utils.data import DataLoader
from collections import defaultdict


begin_time = datetime.datetime.now()

"""
S2FL: Have Your Cake and Eat It Too: Toward Efficient and Accurate Split Federated Learning
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

BatchSizeSever = batchSize * user_parti_num

clip_grad = True
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

user_model, server_model = network.model_selection(cifar, mnist, fmnist, cinic=cinic, split=True, cifar100=cifar100, SVHN=SVHN)

user_model.to(device)
server_model.to(device)

userParam = copy.deepcopy(user_model.state_dict())
serverParam = copy.deepcopy(server_model.state_dict())

optimizer_down = torch.optim.SGD(user_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
optimizer_up = torch.optim.SGD(server_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

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
    def __init__(self, user_data, local_epoch, minibatch=0, computing=0, rate=0, time=0):
        self.user_data = user_data
        self.dataloader_iter = iter(user_data)
        self.local_epoch = local_epoch
        self.count = 0

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

    # Calculation of time
    def model_process(self):
        workload = 5603328  # Workload of one image FLOPs
        # workload *= self.minibatch  # parallel computations
        self.time += (workload / self.computing)

    def transmit_activation(self):
        activation = 131072  # Size of an activation value (bit) (64 * 8 * 8 * 32)
        activation *= self.minibatch
        self.time += (activation / self.rate)

    def transmit_model(self):
        model_volume = 620544
        self.time += (model_volume / self.rate)

# Clients initialization
if WRTT is True:
    clients = [Client(users_data[i], localEpoch, batchSize, clients_computing[i], rates[i], 0) for i in range(user_num)]
else:
    clients = [Client(users_data[i], localEpoch) for i in range(user_num)]

# =========================================================
# ================       S2FL        ======================
# =========================================================

# The workload and activations size for different layers in AlexNet
workloads = [884736,  5603328]
activations_size = [262144,  131072]

def compute_max_time(clients, order, workloads, activations_size, minibatch=32):
    current_time = clients[order[0]].time
    time_vector = []
    clients_time = []
    layers = len(workloads)
    for i in order:
        client_time = []
        for j in range(layers):
            time = 0
            time += (workloads[j] / clients[i].computing) * 2
            time += activations_size[j] / clients[i].rate * minibatch
            time_vector.append(time)
            client_time.append(time)
        clients_time.append(client_time)

    # Calculate the median of the time_vector
    median_time = np.median(time_vector)

    # Find the time closest to the median for each client and record
    closest_times = []
    for client_time in clients_time:
        closest_time = min(client_time, key=lambda x: abs(x - median_time))
        closest_times.append(closest_time)
    max_time = max(closest_times)

    return max_time + current_time

def calculate_l2_distance(label_counts_list, group_indices, num_categories):
    n = num_categories
    total_data_count = 0
    group_label_counts = defaultdict(int)

    for idx in group_indices:
        label_counts = label_counts_list[idx]
        total_data_count += sum(label_counts.values())
        for label, count in label_counts.items():
            group_label_counts[label] += count

    dist = 0
    for i in range(n):
        label_fraction = group_label_counts[i] / total_data_count if total_data_count > 0 else 0
        dist += (label_fraction - 1 / n) ** 2
    return np.sqrt(dist)

# Calculate the number of each label in each DataLoader
def calculate_label_counts(data_loaders, num_categories):
    label_counts_list = []
    for data_loader in data_loaders:
        label_counts = defaultdict(int)
        for _, labels in data_loader:
            for label in labels:
                label_counts[label.item()] += 1
        label_counts_list.append(label_counts)
    return label_counts_list

# Group the data to minimize the L2 distance from a uniform distribution
def group_data(users_data, order, label_counts_list, num_groups=2, num_categories=10):
    num_clients = len(order)
    clients_per_group = num_clients // num_groups

    groups = []
    for i in range(num_groups):
        group = order[i * clients_per_group:(i + 1) * clients_per_group].tolist()
        groups.append(group)

    def calculate_average_distance(groups):
        total_distance = 0
        for group in groups:
            total_distance += calculate_l2_distance(label_counts_list, group, num_categories)
        return total_distance / num_groups

    # print(calculate_average_distance(groups))

    # Optimize the grouping
    for _ in range(100):
        for group_index, group in enumerate(groups):
            for client_index in list(group):
                current_average_distance = calculate_average_distance(groups)
                best_groups = groups
                best_average_distance = current_average_distance

                for other_group_index, other_group in enumerate(groups):
                    if other_group_index != group_index:
                        # Attempt to swap clients between two groups
                        for other_client_index in list(other_group):
                            new_groups = [grp.copy() for grp in groups]
                            # print(new_groups[group_index])
                            # print(client_index)
                            if client_index not in new_groups[group_index]:
                                break
                            new_groups[group_index].remove(client_index)
                            new_groups[other_group_index].remove(other_client_index)
                            new_groups[group_index].append(other_client_index)
                            new_groups[other_group_index].append(client_index)
                            new_average_distance = calculate_average_distance(new_groups)

                            if new_average_distance < best_average_distance:
                                best_groups = new_groups
                                best_average_distance = new_average_distance

                # If a better grouping is found, update the groups
                if best_groups != groups:
                    groups = best_groups
                    break
    # print(best_average_distance)
    return groups


# =========================================================
# ================       training        ==================
# =========================================================

def modify_dataloader_batch_size(dataloader, new_batch_size):
    return DataLoader(dataset=dataloader.dataset,
                     batch_size=new_batch_size,
                     shuffle=True,
                     drop_last=dataloader.drop_last)

def find_max_time(clients, order):
    max_time = 0
    for index in order:
        client = clients[index]
        if client.time > max_time:
            max_time = client.time
    return max_time

def update_all_clients_time(clients, order, time):
    for index in order:
        clients[index].time = time

total_accuracy = []
total_loss = []
time_record = []
current_time = 0

# Calculate the label count for each client's data
label_counts_list = calculate_label_counts(users_data, num_categories=classOfLabel)

for epoch in range(epochs):
    user_model.train()
    server_model.train()
    avgloss = 0

    '''Randomly select user_parti_num users'''
    order = np.random.choice(range(user_num), user_parti_num, replace=False)
    print('FL select clients:', order)
    if WRTT:  # Initialize the timeline for selected clients
        update_all_clients_time(clients, order, current_time)

    '''Calculate aggregation weights'''
    total_data_points = sum([len(users_data[r].dataset) for r in order])
    fed_avg_freqs = [len(users_data[r].dataset) / total_data_points for r in order]
    # print(fed_avg_freqs)

    '''Modify the batch size of the client dataloader'''
    index = 0
    for i in order:
        new_batch_size = max(int(fed_avg_freqs[index] * BatchSizeSever), 1)
        users_data[i] = modify_dataloader_batch_size(users_data[i], new_batch_size)
        index += 1

    usersParam = []
    for i in range(user_parti_num):
        usersParam.append(copy.deepcopy(userParam))

    groups = group_data(users_data, order, label_counts_list, num_groups=2, num_categories=10)
    for j in range(localEpoch):
        if WRTT:
            max_time = compute_max_time(clients, order, workloads, activations_size)
            update_all_clients_time(clients, order, max_time)

        sumServerParam = None
        for group in groups:
            concat_features = None
            concat_labels = None
            for client_index in group:
                user_model.load_state_dict(copy.deepcopy(usersParam[np.where(order == client_index)[0][0]]), strict=True)
                server_model.load_state_dict(copy.deepcopy(serverParam), strict=True)

                images, labels = clients[client_index].train_one_iteration()
                if concat_labels == None:
                    concat_labels = labels
                else:
                    concat_labels = torch.cat((concat_labels, labels), dim=0)
                images = images.to(device)
                labels = labels.to(device)
                split_layer_output = user_model(images)

                '''concatenation'''
                if concat_features == None:
                    concat_features = split_layer_output.detach()
                else:
                    concat_features = torch.cat((concat_features, split_layer_output.detach()), dim=0)

                local_output = server_model(split_layer_output)
                localLoss = criterion(local_output, labels.long())
                optimizer_down.zero_grad()
                localLoss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(parameters=user_model.parameters(), max_norm=10)

                optimizer_down.step()

                usersParam[np.where(order == client_index)[0][0]] = copy.deepcopy(user_model.state_dict())

            '''server-side model update'''
            concat_labels = concat_labels.to(device)
            server_model.load_state_dict(copy.deepcopy(serverParam), strict=True)
            final_output = server_model(concat_features)
            loss = criterion(final_output, concat_labels.long())
            optimizer_up.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(parameters=server_model.parameters(), max_norm=10)

            optimizer_up.step()

            if sumServerParam is None:
                sumServerParam = server_model.state_dict()
                for key in server_model.state_dict():
                    sumServerParam[key] = server_model.state_dict()[key].clone() * (1 / len(groups))
            else:
                for key in server_model.state_dict():
                    sumServerParam[key] += server_model.state_dict()[key].clone() * (1 / len(groups))
        '''server-side model aggregation'''
        serverParam = copy.deepcopy(sumServerParam)

    '''client-side model aggregation'''
    test_flag = ((epoch + 1) % Accu_Test_Frequency == 0)
    if WRTT:
        for i in order:
            clients[i].transmit_model()
        current_time = find_max_time(clients, order)
        if test_flag:
            time_record.append(current_time)
            print("Time: " + str(current_time))
    sumClientParam = None
    for i in range(user_parti_num):
        if sumClientParam == None:
            sumClientParam = usersParam[i]
            for key in usersParam[i]:
                sumClientParam[key] = usersParam[i][key] * fed_avg_freqs[i]
        else:
            for key in usersParam[i]:
                sumClientParam[key] += usersParam[i][key] * fed_avg_freqs[i]
    userParam = copy.deepcopy(sumClientParam)

    avgloss = avgloss / user_parti_num
    total_loss.append(avgloss)

# =========================================================
# ============       testing per round        =============
# =========================================================
    if test_flag:
        user_model.eval()
        server_model.eval()
        server_model.load_state_dict(serverParam, strict=True)
        user_model.load_state_dict(userParam, strict=True)

        with torch.no_grad():
            correct = 0
            total = 0
            for (images, labels) in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = user_model(images)
                output = server_model(output)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuary = correct / total
        total_accuracy.append(accuary)
        print("Global iteration: " + str(epoch + 1))
        print("Accuracy: ", total_accuracy[-1])
        print()

print(time_record)
print(total_accuracy)
time_record_str = ', '.join(str(x) for x in time_record)
total_accuracy_str = ', '.join(str(x) for x in total_accuracy)
print('time = [' + time_record_str + ']')
print('S2FL = [' + total_accuracy_str + ']')
print(clip_grad)

end_time = datetime.datetime.now()

begin_time_str = begin_time.strftime("%Y-%m-%d %H:%M:%S")
end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

selectDataset = "cifar10" if cifar else "mnist" if mnist else "fmnist" if fmnist else "cinic" if cinic else "cifar100" \
    if cifar100 else "SVHN" if SVHN else "None"

IfCilp = "clip" if clip_grad else "not clip"


with open('S2FL.txt', 'w') as f:
    if dirichlet:
        f.write(
            f'seed_value: {seed_value}; alpha: {alpha}; epochs: {epoches}; {selectDataset}; local epoch: {localEpoch}; {IfCilp};\n'
            f'num of clients: {user_num}; num of participating clients: {user_parti_num}; batchsize: {batchSize}; learning rate: {lr}; \n')
    else:
        f.write(
            f'seed_value: {seed_value}; shard: {shard}; epochs: {epoches}; {selectDataset}; local epoch: {localEpoch}; {IfCilp};\n'
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
    f.write('S2FL = [' + total_accuracy_str + ']\n')


