from utils.plot_utils import *
from simulate import simulate
from data.Femnist.data_generator import generate_data as femnist_generator


def create_dataset(dataset, total_users, similarity, samples_num):
    if dataset == "Femnist":
        femnist_generator(similarity, total_users, samples_num)
    if dataset == "CIFAR-10":
        pass  # TODO : create data generator for CIFAR-10
    if dataset == "Mnist":
        pass  # TODO : create data generator for Mnist


input_dict = {"dataset": "CIFAR-10",
              "algorithm": None,
              "model": "CIFAR-10",
              "num_glob_iters": 50,
              "batch_size": 60,
              "learning_rate": 0.008,
              "local_epochs": None,
              "L": 0.004,
              "users_per_round": 0,
              "similarity": 1,
              "times": 1,
              "noise": False}

algorithms = ["SCAFFOLD", "FedAvg"]
epochs = [1]

for alg in algorithms:
    for ep in epochs:
        input_dict["algorithm"] = alg
        input_dict["local_epochs"] = ep
        simulate(hyper_learning_rate=0, optimizer=None, rho=0, **input_dict)

# for similarity in similarities:
#     create_dataset(input_dict["dataset"], 100, similarity, 20)
#     input_dict["similarity"] = similarity
#     for alg in algorithms:
#         for ep in epochs:
#             input_dict["algorithm"] = alg
#             input_dict["local_epochs"] = ep
#             simulate(hyper_learning_rate=0, L=0, optimizer=None, rho=0, **input_dict)

plot_dict = get_plot_dict(input_dict, algorithms, epochs)
plot_by_epochs(**plot_dict)
