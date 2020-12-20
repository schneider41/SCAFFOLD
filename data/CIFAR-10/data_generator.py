import emnist
import numpy as np
from tqdm import trange
import random
import json
import os
from pickle import Unpickler

train_path = './data/train/mytrain.json'
test_path = './data/test/mytest.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
    return data_dict


cifar_dicts = []
for i in range(1, 6):
    cifar_dicts.append(unpickle('./cifar-10-batches-py/data_batch_' + f"{i}"))

train_images = np.concatenate([cifar_dict['data'] for cifar_dict in cifar_dicts])
train_labels = np.concatenate([cifar_dict['labels'] for cifar_dict in cifar_dicts])
train_images = train_images.astype(np.float32)
train_labels = train_labels.astype(np.int)
num_of_labels = len(set(train_labels))

cifar_data = []
for i in range(min(train_labels), num_of_labels + min(train_labels)):
    idx = train_labels == i
    cifar_data.append(train_images[idx])

# Assign 100 samples to each user
similarity = 0.8  # between 0 to 1
num_of_users = 10
samples_num = 5000
iid_samples = int(similarity * samples_num)
X = [[] for _ in range(num_of_users)]
y = [[] for _ in range(num_of_users)]
idx = np.zeros(num_of_labels, dtype=np.int64)

# fill users data by labels
for user in range(num_of_users):
    label = user % num_of_labels
    X[user] += cifar_data[label][idx[label]:idx[label] + samples_num - iid_samples].tolist()
    y[user] += (label * np.ones(samples_num - iid_samples)).tolist()
    idx[label] += samples_num - iid_samples

print(idx)

# create %similarity of iid data
for user in range(num_of_users):
    labels = np.random.randint(0, num_of_labels, iid_samples)
    for label in labels:
        while idx[label] >= len(cifar_data[label]):
            label = (label + 1) % num_of_labels
        X[user].append(cifar_data[label][idx[label]].tolist())
        y[user] += (label * np.ones(1)).tolist()
        idx[label] += 1

print(idx)

train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

for i in trange(num_of_users, ncols=120):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.9 * num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print(train_data['num_samples'])
print(sum(train_data['num_samples']))

print("Saving data, please wait")
with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
print("Saving completed")
