from utils.plot_utils import *
from simulate import simulate

"start simulation to our choosing"
"1st simulation written by ram- check FedAvg vs Scaffold with AWGN"

input_dict = {"dataset": "CIFAR-10",
              "algorithm": "SCAFFOLD",
              "model": "CIFAR-10",
              "num_glob_iters": 250,
              "batch_size": 60,
              "learning_rate": 0.01,
              "local_epochs": 1,
              "users_per_round": 0,
              "similarity": 0.8,
              "times": 1}

algorithms = ["FedAvg"]
epochs = [1]

# input_dict["algorithm"] = algorithms[1]
# input_dict["local_epochs"] = 1
# simulate(**input_dict, hyper_learning_rate=0, L=0, optimizer=None, rho=0)

for alg in algorithms:
    for ep in epochs:
        input_dict["algorithm"] = alg
        input_dict["local_epochs"] = ep
        simulate(hyper_learning_rate=0, L=0, optimizer=None, rho=0, **input_dict)

plot_dict = get_plot_dict(input_dict, algorithms, epochs)
plot_by_epochs(**plot_dict)



