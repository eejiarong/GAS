import sys
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
utils_path = os.path.join(parent_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

import copy
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import random
import numpy as np
import datetime
from network import model_selection
from dataset import Dataset, Data_Partition
from utils import calculate_v_value, replace_user, sample_or_generate_features, \
    compute_local_adjustment, find_client_with_min_time


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
clip_grad = True

# Hyperparameter Setting of GAS
Generate = True  # Whether to generate activations
Sample_Frequency = 1  # Sampling frequency
V_Test = False  # Calculate Gradient Dissimilarity
V_Test_Frequency = 1
Accu_Test_Frequency = 1
num_label = 100 if cifar100 else 10

# Simulate real communication environments
WRTT = True   # True for simulation, False for no simulation


'''communication formulation'''


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


'''Class Definition'''


# Client class definition
class Client:
    def __init__(self, user_data, local_epoch, minibatch=0, computing=0, rate=0, time=0, weight_count=1):
        self.user_data = user_data
        self.dataloader_iter = iter(user_data)
        self.local_epoch = local_epoch
        self.count = 0
        # Calculation of time
        self.minibatch = minibatch
        self.computing = computing
        self.rate = rate
        self.time = time
        # weight
        self.weight_count = weight_count

    def increment_counter(self):
        # record the number of local iterations
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
        # AlexNet
        workload = 5603328  # Workload of one image FLOPs
        # workload *= self.minibatch  # parallel computations
        self.time += (workload / self.computing)

    def transmit_activation(self):
        # AlexNet
        activation = 131072  # Size of an activation value (bit) (64 * 8 * 8 * 32)
        activation *= self.minibatch
        self.time += (activation / self.rate)

    def transmit_model(self):
        # AlexNet
        model_volume = 620544
        self.time += (model_volume / self.rate)


# IncrementalStats class for maintaining mean and variance
class IncrementalStats:
    def __init__(self, device):
        self.device = device
        self.means = {}
        self.variances = {}
        self.weight = {}
        self.counts = {}

    def update(self, new_mean, new_cov, new_weight, label):
        """
        Update the weighted mean and variance of the features for the given label.
        :param feature: Feature vector of type torch.Tensor
        :param label: Label
        """
        n = new_mean.shape[0]
        regularization_term = 1e-5
        I = torch.eye(n).to(self.device)

        if label not in self.means:
            self.means[label] = new_mean.to(self.device)
            self.variances[label] = new_cov.to(self.device)
            self.counts[label] = 1
            self.weight[label] = new_weight
        else:
            old_mean = self.means[label]
            old_cov = self.variances[label]
            old_weight = self.weight[label]
            self.weight[label] = old_weight + new_weight
            decay_factor = old_weight / self.weight[label]

            self.means[label] = decay_factor * old_mean + (1 - decay_factor) * new_mean
            self.variances[label] = decay_factor * (
                    old_cov + torch.outer(self.means[label] - old_mean, self.means[label] - old_mean)) \
                                    + (1 - decay_factor) * (new_cov + torch.outer(self.means[label] - new_mean,
                                                                                  self.means[
                                                                                      label] - new_mean)) + regularization_term * I
            self.counts[label] += 1

            ''' If the activation size is large and the covariance matrix of the activation values
                occupies a significant amount of memory, approximate the covariance matrix using a diagonal matrix.'''
            # old_mean = self.means[label]
            # old_var = self.variances[label]
            # old_weight = self.weight[label]
            # new_var = new_cov
            # self.weight[label] = old_weight + new_weight
            # decay_factor = old_weight / self.weight[label]
            # self.means[label] = decay_factor * old_mean + (1 - decay_factor) * new_mean
            # self.variances[label] = decay_factor * (
            #         old_var + (self.means[label] - old_mean) ** 2
            # ) + (1 - decay_factor) * (
            #                                 new_var + (self.means[label] - new_mean) ** 2
            #                         ) + regularization_term
            # self.counts[label] += 1


    def get_stats(self, label):
        """
        Get the weighted mean and variance for the given label.
        :param label: Label
        :return: Weighted mean and variance of type torch.Tensor
        """
        return self.means.get(label, None), self.variances.get(label, None)


'''Main Train'''

# Data loading and preprocessing
alldata, alllabel, test_set, transform = Dataset(cifar=cifar, mnist=mnist, fmnist=fmnist, cinic=cinic,
                                                 cifar100=cifar100, SVHN=SVHN)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=128, shuffle=True)
train_index = np.arange(0, len(alldata))
random.shuffle(train_index)
train_img = np.array(alldata)[train_index]
train_label = np.array(alllabel)[train_index]
users_data = Data_Partition(iid, dirichlet, train_img, train_label, transform, user_num, batchSize, alpha, shard,
                            drop=False, classOfLabel=num_label)

# Model initialization
user_model, server_model = model_selection(cifar, mnist, fmnist, cinic=cinic, split=True, cifar100=cifar100, SVHN=SVHN)
user_model.to(device)
server_model.to(device)
userParam = copy.deepcopy(user_model.state_dict())
optimizer_down = torch.optim.SGD(user_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
optimizer_up = torch.optim.SGD(server_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

# Initialize clients
if WRTT is True:
    clients = [Client(users_data[i], localEpoch, batchSize, clients_computing[i], rates[i], 0) for i in range(user_num)]
else:
    clients = [Client(users_data[i], localEpoch) for i in range(user_num)]
stats = IncrementalStats(device=device)
condensed_data = {c: None for c in range(num_label)}
train_begin_time = datetime.datetime.now()

# Training loop
total_accuracy = []
total_v_value = []
local_models_time = []
time_record = []
epoch = 0
order = np.random.choice(range(user_num), user_parti_num, replace=False)  # 初始选择的用户
if WRTT is True:  # initialize training time
    for i in order:
        clients[i].model_process()
        clients[i].transmit_activation()
usersParam = [copy.deepcopy(userParam) for _ in range(user_parti_num)]
concat_features = None
concat_labels = None
concat_weight_counts = None
sumClientParam = None
feature_shape = None
count_concat = 0  # activation cache
count_local = 0  # local-side model cache
local_epoch = 0  # total number of local iterations
total_weight_count = 1

# generate local logit adjustment for each client
logit_local_adjustments = []
for i in range(user_num):
    logit_local_adjustments.append(compute_local_adjustment(users_data[i], device))

while epoch != epochs:
    user_model.train()
    server_model.train()
    # select a client
    if WRTT is True:
        selected_client = find_client_with_min_time(clients, order)
    else:
        selected_client = np.random.choice(order)
    user_model.load_state_dict(usersParam[np.where(order == selected_client)[0][0]], strict=True)
    # train
    images, labels = clients[selected_client].train_one_iteration()
    images = images.to(device)
    labels = labels.to(device)
    split_layer_output = user_model(images)
    if feature_shape is None:
        feature_shape = split_layer_output[0].shape

    # Define the weight vector to record the weight of each activation value
    weight_count = clients[selected_client].weight_count
    weight_vector = torch.tensor([weight_count] * split_layer_output.size(0))

    # generate concatenated activation
    features = split_layer_output.detach()
    count_concat += 1
    if concat_features is None:
        concat_features = features
        concat_labels = labels
        concat_weight_counts = weight_vector
    else:
        concat_features = torch.cat((concat_features, split_layer_output.detach()), dim=0)
        concat_labels = torch.cat((concat_labels, labels), dim=0)
        concat_weight_counts = torch.cat((concat_weight_counts, weight_vector), dim=0)

    # Update weight of clients
    clients[selected_client].weight_count = clients[selected_client].weight_count + 1

    # client-side model update
    local_output = server_model(split_layer_output)

    # localLoss = criterion(local_output, labels)
    localLoss = criterion(local_output + logit_local_adjustments[selected_client], labels.long())
    optimizer_down.zero_grad()
    localLoss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(parameters=user_model.parameters(), max_norm=10)
    optimizer_down.step()
    usersParam[np.where(order == selected_client)[0][0]] = copy.deepcopy(user_model.state_dict())
    if WRTT is True:  # Record the time of the backward pass
        clients[selected_client].model_process()

    '''Activations generation and server-side model update'''
    if count_concat == user_parti_num:
        # print("local_epoch: " + str(local_epoch))
        local_epoch += 1
        # update activation distributions
        unique_labels, counts = concat_labels.unique(return_counts=True)  # Count how many of each label

        label_weights = {}
        concat_weight_counts = concat_weight_counts.to(device)
        for label in unique_labels:
            mask = (concat_labels == label)
            weights_of_label = concat_weight_counts[mask].float()
            label_weights[label.item()] = weights_of_label.sum().item()

        # Calculate mean and variance
        flatten_features = concat_features.flatten(start_dim=1)
        for label in unique_labels:
            mask = (concat_labels == label)
            features_of_label = flatten_features[mask]
            weights_of_label = concat_weight_counts[mask].float()
            total_weight = weights_of_label.sum()
            mean_feature = torch.sum(features_of_label * weights_of_label[:, None], dim=0) / total_weight
            centered_features = features_of_label - mean_feature
            cov_matrix = torch.matmul((centered_features * weights_of_label[:, None]).T,
                                      centered_features) / total_weight
            stats.update(mean_feature, cov_matrix, label_weights[label.item()], label.item())
            ''' If the activation size is large and the covariance matrix of the activation values
                occupies a significant amount of memory, approximate the covariance matrix using a diagonal matrix.'''
            # var_vector = torch.sum((centered_features ** 2) * weights_of_label[:, None], dim=0) / total_weight
            # stats.update(mean_feature, var_vector, label_weights[label.item()], label.item())
        if Generate is True:
            # Activations generation
            if local_epoch % Sample_Frequency == 0:
                # Ensure that all labels have mean and variance
                all_labels_have_stats = True
                for label in range(num_label):
                    if stats.get_stats(label) == (None, None):
                        all_labels_have_stats = False
                        break
                if all_labels_have_stats:
                    concat_features, concat_labels = sample_or_generate_features(concat_features, concat_labels,
                                                                                 batchSize, num_label, feature_shape,
                                                                                 device, stats)

        # server-side model update
        for param in server_model.parameters():
            param.requires_grad = True
        final_output = server_model(concat_features)
        loss = criterion(final_output, concat_labels.long())
        optimizer_up.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(parameters=server_model.parameters(), max_norm=10)
        optimizer_up.step()
        if V_Test is True:
            concat_labels_V = copy.deepcopy(concat_labels)
            concat_features_V = copy.deepcopy(concat_features)
        count_concat = 0
        concat_labels = None
        concat_features = None
        concat_weight_counts = None

    # client-side models aggregation
    replace = clients[selected_client].increment_counter()
    if replace:  # If local iterations are completed, select a new client
        count_local += 1
        if WRTT is True:  # Record the time of model upload
            clients[selected_client].transmit_model()
            local_models_time.append(clients[selected_client].time)
        if sumClientParam is None:
            sumClientParam = usersParam[np.where(order == selected_client)[0][0]]
            for key in usersParam[np.where(order == selected_client)[0][0]]:
                sumClientParam[key] = usersParam[np.where(order == selected_client)[0][0]][key] * (1 / user_parti_num)
        else:
            for key in usersParam[np.where(order == selected_client)[0][0]]:
                sumClientParam[key] += usersParam[np.where(order == selected_client)[0][0]][key] * (1 / user_parti_num)
        if count_local == user_parti_num:  # Update the client model if the buffer is full
            total_weight_count += local_epoch
            userParam = copy.deepcopy(sumClientParam)
            sumClientParam = None
            count_local = 0

            test_flag = ((epoch + 1) % Accu_Test_Frequency == 0)
            epoch += 1

            if WRTT:
                if test_flag:
                    time_record.append(max(local_models_time))
                    print("Time: " + str(time_record[-1]))
                local_models_time = []

            # Accuracy test
            if test_flag:
                user_model.eval()
                server_model.eval()
                user_model.load_state_dict(userParam, strict=True)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = user_model(images)
                        output = server_model(output)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                total_accuracy.append(accuracy)
                print("Global iteration: " + str(epoch))
                print("Accuracy: " + str(total_accuracy[-1]))
                print()

            # V test
            if V_Test and (epoch + 1) % V_Test_Frequency == 0:
                v_value = calculate_v_value(server_model, user_model, concat_features_V, concat_labels_V, test_loader,
                                            criterion, device)
                print(f"Epoch {epoch + 1}, V Value: {v_value}")
                total_v_value.append(v_value)

        # select new client
        index = np.where(order == selected_client)[0][0]
        usersParam[index] = userParam

        if WRTT is True:  # Initialize the time for the new client
            begin_time = clients[selected_client].time
            order = replace_user(order, selected_client, user_num)
            clients[order[index]].weight_count = total_weight_count
            clients[order[index]].time = begin_time
            clients[order[index]].model_process()
            clients[order[index]].transmit_activation()
        else:
            order = replace_user(order, selected_client, user_num)
            clients[order[index]].weight_count = total_weight_count
    else:
        if WRTT is True:  # Record the training time if the client continues with local iterations
            clients[selected_client].model_process()
            clients[selected_client].transmit_activation()

# Output results
print(time_record)
print(total_accuracy)
print(total_v_value)
time_record_str = ', '.join(str(x) for x in time_record)
total_accuracy_str = ', '.join(str(x) for x in total_accuracy)
total_v_value_str = ', '.join(str(x) for x in total_v_value)
print('time = [' + time_record_str + ']')
print('GAS = [' + total_accuracy_str + ']')

end_time = datetime.datetime.now()
begin_time_str = train_begin_time.strftime("%Y-%m-%d %H:%M:%S")
end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

selectDataset = "cifar10" if cifar else "mnist" if mnist else "fmnist" if fmnist else "cinic" if cinic else "cifar100" \
    if cifar100 else "SVHN" if SVHN else "None"
selectMethod = "Generative Activation-Aided" if Generate else "Original"
IfCilp = "clip" if clip_grad else "not clip"

with open('GAS_main.txt', 'w') as f:
    if dirichlet:
        f.write(
            f'seed_value: {seed_value}; alpha: {alpha}; epochs: {epochs}; {selectDataset}; local epoch: {localEpoch}; {IfCilp};\n'
            f'num of clients: {user_num}; num of participating clients: {user_parti_num}; batchsize: {batchSize}; learning rate: {lr}; \n'
            f'Method: {selectMethod}; sample frequency: {Sample_Frequency}; \n')
    else:
        f.write(
            f'seed_value: {seed_value}; shard: {shard}; epochs: {epochs}; {selectDataset}; local epoch: {localEpoch}; {IfCilp};\n'
            f'num of clients: {user_num}; num of participating clients: {user_parti_num}; batchsize: {batchSize}; learning rate: {lr}; \n'
            f'Method: {selectMethod}; sample frequency: {Sample_Frequency}; \n')
    if V_Test is True:
        f.write(f'Test Frequency is {V_Test_Frequency}; \n')
        f.write('Gradient Dissimilarity = [' + total_v_value_str + ']\n')

    if WRTT is True:
        clients_computing_str = ', '.join(str(x) for x in clients_computing)
        clients_position_str = ', '.join(str(x) for x in clients_position)
        rates_str = ', '.join(str(x) for x in rates)
        f.write('clients computing = [' + clients_computing_str + ']\n')
        f.write('clients position = [' + clients_position_str + ']\n')
        f.write('clients rates = [' + rates_str + ']\n')
        f.write('time = [' + time_record_str + ']\n')

    f.write(f'{begin_time_str} ~ {end_time_str};\n')
    f.write('GAS = [' + total_accuracy_str + ']\n')


