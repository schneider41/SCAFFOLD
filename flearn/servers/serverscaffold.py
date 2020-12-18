import torch
import os

import h5py
from flearn.users.userscaffold import UserSCAFFOLD
from flearn.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
from scipy.stats import rayleigh


# Implementation for SCAFFOLD Server
class SCAFFOLD(Server):
    def __init__(self, dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
                 local_epochs, optimizer, users_per_round, similarity, times):
        super().__init__(dataset, algorithm, model[0], batch_size, learning_rate, hyper_learning_rate, L,
                         num_glob_iters, local_epochs, optimizer, users_per_round, 0, similarity, times)

        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.communication_thresh = rayleigh.ppf(0.2)  # h_min

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)
            user = UserSCAFFOLD(id, train, test, model, batch_size, learning_rate, hyper_learning_rate, L, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating SCAFFOLD server.")

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0

            self.send_parameters()

            # Evaluate model at each iteration
            self.evaluate()

            self.selected_users = self.select_users(glob_iter, self.users_per_round)
            for user in self.selected_users:
                user.train(self.local_epochs)
            # self.selected_users = self.select_transmitting_users()
            # print(f"Transmitting {len(self.selected_users)} users")
            self.aggregate_parameters()
            self.apply_channel_effect()
            # loss_ /= self.total_train_samples
            # loss.append(loss_)
            # print(loss_)
        # print(loss)
        self.save_results()
        self.save_model()

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
            for control, new_control in zip(user.server_controls, self.server_controls):
                control.data = new_control.data

    def select_transmitting_users(self):
        transmitting_users = []
        for user in self.selected_users:
            user.csi = rayleigh.rvs()
            if user.csi >= self.communication_thresh:
                transmitting_users.append(user)
        return transmitting_users

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, total_train)

    def add_parameters(self, user, total_samples):
        num_of_selected_users = len(self.selected_users)
        num_of_users = len(self.users)
        num_of_samples = user.train_samples
        for param, control, del_control, del_model in zip(self.model.parameters(), self.server_controls,
                                                          user.delta_controls, user.delta_model):
            param.data = param.data + del_model.data * num_of_samples / num_of_selected_users
            # control.data = control.data + del_control.data * num_of_samples / total_samples
            # param.data = param.data + del_model.data / num_of_selected_users
            control.data = control.data + del_control.data / num_of_users

    def apply_channel_effect(self, sigma=1, power_control=1e26):
        num_of_selected_users = len(self.selected_users)
        users_norms = []
        for user in self.selected_users:
            users_norms.append(user.get_params_norm())
        alpha_t = power_control / max(users_norms) ** 2
        alpha_t = 1000
        for param, control in zip(self.model.parameters(), self.server_controls):
            param.data = param.data + 1 / (alpha_t ** 0.5 * num_of_selected_users * self.communication_thresh) * sigma * torch.randn(param.data.size())
            control.data = control.data + 1 / (alpha_t ** 0.5 * num_of_selected_users * self.communication_thresh) * sigma * torch.randn(control.data.size())
