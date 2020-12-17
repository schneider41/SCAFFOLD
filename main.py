from utils.plot_utils import *
from simulate import simulate

"start simulation to our choosing"
"1st simulation written by ram- check FedAvg vs Scaffold with AWGN"

input_dict = {"dataset": "Femnist",
              "num_of_users": 100,
              "algorithm": "SCAFFOLD",
              "model": "mclr",
              "num_glob_iters": 250,
              "batch_size": 4,
              "learning_rate": 0.01,
              "local_epochs": 1,
              "clients_per_round": 20,
              "similarity": 1,
              "times": 1}

algorithms = ["FedAvg", "SCAFFOLD"]
epochs = [1, 5]

# input_dict["algorithm"] = algorithms[1]
# input_dict["local_epochs"] = 1
# simulate(**input_dict, hyper_learning_rate=0, L=0, optimizer=None, rho=0)

for alg in algorithms:
    for ep in epochs:
        input_dict["algorithm"] = alg
        input_dict["local_epochs"] = ep
        simulate(**input_dict, hyper_learning_rate=0, L=0, optimizer=None, rho=0)

plot_dict = get_plot_dict(input_dict, algorithms, epochs)
plot_by_epochs(**plot_dict)



