#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from flearn.servers.serveravg import FedAvg
from flearn.servers.serverfedl import FEDL
from flearn.servers.serverscaffold import SCAFFOLD
from flearn.trainmodel.models import *
from utils.plot_utils import *
import torch

torch.manual_seed(0)


def simulate(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
             local_epochs, optimizer, users_per_round, rho, times, similarity):
    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm: {algorithm}")
    print(f"Batch size              : {batch_size}")
    print(f"Learing rate            : {learning_rate}")
    print(f"Subset of users         : {users_per_round}")
    print(f"Number of local rounds  : {local_epochs}")
    print(f"Number of global rounds : {num_glob_iters}")
    print(f"Dataset                 : {dataset}")
    print(f"Data Similarity         : {similarity}")
    print(f"Local Model             : {model}")
    print("=" * 80)

    for i in range(times):
        print("---------------Running time:------------", i)

        # Generate model
        if model == "mclr":  # for Mnist and Femnist datasets
            model = MclrLogistic(output_dim=47), model

        if model == "linear":  # For Linear dataset
            model = LinearRegression(40, 1), model

        if model == "dnn":  # for Mnist and Femnist datasets
            model = DNN(), model

        if model == "CIFAR-10":  # for Cifar-10 dataset
            model = CifarNet(), model

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L,
                            num_glob_iters, local_epochs, optimizer, users_per_round, rho, similarity, i)

        if algorithm == "FEDL":
            server = FEDL(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
                          local_epochs, optimizer, users_per_round, rho, i)

        if algorithm == "SCAFFOLD":
            server = SCAFFOLD(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L,
                              num_glob_iters, local_epochs, optimizer, users_per_round, similarity, i)
        server.train()
        server.test()

    # Average data
    average_data(clients_per_round=users_per_round, local_epochs=local_epochs, num_glob_iters=num_glob_iters,
                 learning_rate=learning_rate, algorithm=algorithm, batch_size=batch_size, dataset=dataset, times=times,
                 similarity=similarity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                        choices=["CIFAR-10", "Mnist", "CIFAR-10", "Linear_synthetic", "Logistic_synthetic"])
    parser.add_argument("--similarity", type=int, default=0.8)
    parser.add_argument("--model", type=str, default="CIFAR-10", choices=["linear", "mclr", "dnn", "CIFAR-10"])
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--hyper_learning_rate", type=float, default=0.02, help=" Learning rate of FEDL")
    parser.add_argument("--L", type=int, default=0, help="Regularization term")
    parser.add_argument("--num_glob_iters", type=int, default=400)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="")
    parser.add_argument("--algorithm", type=str, default="FedAvg", choices=["FEDL", "FedAvg", "SCAFFOLD"])
    parser.add_argument("--clients_per_round", type=int, default=0, help="Number of Users per round")
    parser.add_argument("--rho", type=float, default=0, help="Condition Number")
    parser.add_argument("--times", type=int, default=1, help="running time")
    args = parser.parse_args()

    simulate(dataset=args.dataset, algorithm=args.algorithm, model=args.model,
             batch_size=args.batch_size, learning_rate=args.learning_rate, hyper_learning_rate=args.hyper_learning_rate,
             L=args.L, num_glob_iters=args.num_glob_iters, local_epochs=args.local_epochs, optimizer=args.optimizer,
             users_per_round=args.clients_per_round, rho=args.rho, times=args.times, similarity=args.similarity)
