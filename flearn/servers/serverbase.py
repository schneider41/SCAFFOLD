import torch
import os
import h5py

import numpy as np
from utils.model_utils import Metrics
import copy


class Server:
    def __init__(self, dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L,
                 num_glob_iters, local_epochs, optimizer, users_per_round, rho,similarity,times):

        # Set up the main attributes
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.users_per_round = users_per_round
        self.hyper_learning_rate = hyper_learning_rate
        self.L = L
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []

        self.rho = rho
        self.times = times
        self.similarity = similarity

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio
            if user_param.grad != None:
                if server_param.grad == None:
                    server_param.grad = torch.zeros_like(user_param.grad)
                server_param.grad.data = server_param.grad.data + user_param.grad.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
            if param.grad != None:
                param.grad.data = torch.zeros_like(param.grad.data)
        total_train = 0
        # if(self.users_per_round = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_grad(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, users_per_round):
        if users_per_round in [len(self.users), 0]:
            return self.users

        users_per_round = min(users_per_round, len(self.users))
        # fix the list of user consistent
        np.random.seed(round * (self.times + 1))
        return np.random.choice(self.users, users_per_round, replace=False)  # , p=pk)

    def save_results(self):
        """ Save loss, accuracy to h5 file"""
        file_name = "./results/" + self.dataset + "_" + self.algorithm
        file_name += "_" + str(self.learning_rate) + "lr"
        file_name += "_" + str(self.users_per_round) + "u"
        file_name += "_" + str(self.batch_size) + "b"
        file_name += "_" + str(self.local_epochs) + "e"
        file_name += "_" + str(self.similarity) + "s"
        file_name += "_" + str(self.times) + ".h5"
        if len(self.rs_glob_acc) != 0 & len(self.rs_train_acc) & len(self.rs_train_loss):
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)

