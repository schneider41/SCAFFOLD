import emnist
import numpy as np
from tqdm import trange
import random
import json
import os

train_path = './data/train/mytrain.json'
test_path = './data/test/mytest.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dataset = 'balanced'
images, labels = emnist.extract_training_samples(dataset) # TODO: add test samples
images = np.reshape(images, (images.shape[0], -1))
images = images.astype(np.float32)
labels = labels.astype(np.int)
num_of_labels = len(set(labels))


emnist_data = []
for i in range(min(labels), num_of_labels + min(labels)):
    idx = labels == i
    emnist_data.append(images[idx])


# Assign 100 samples to each user
similarity = 0  # between 0 to 1
num_of_clients = 100
samples_num = 20
similar_data = int(similarity * samples_num)
X = [[] for _ in range(num_of_clients)]
y = [[] for _ in range(num_of_clients)]
idx = np.zeros(num_of_labels, dtype=np.int64)

# create %similarity of iid data
for user in range(num_of_clients):
    data = np.random.randint(0, num_of_labels, similar_data)
    for label in data:
        X[user].append(emnist_data[label][idx[label]].tolist())
        y[user] += (label * np.ones(1)).tolist()
        idx[label] += 1

print(idx)

# fill remaining data
for user in range(num_of_clients):
    label = user % num_of_labels
    X[user] += emnist_data[label][idx[label]:idx[label] + samples_num - similar_data].tolist()
    y[user] += (label * np.ones(samples_num - similar_data)).tolist()
    idx[label] += samples_num - similar_data

print(idx)

# Create data structure

train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Setup 100 users
for i in trange(num_of_clients, ncols=120):
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

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)