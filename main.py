from utils.plot_utils import *
from simulate import simulate
from data.Femnist.data_generator import generate_data as generate_femnist_data

input_dict = {"dataset": "Femnist",
              "algorithm": None,
              "model": "mclr",
              "num_glob_iters": 300,
              "batch_size": 4,
              "learning_rate": 0.001,
              "local_epochs": 1,
              "L": 0,
              "users_per_round": 20,
              "similarity": None,
              "times": 1,
              "noise": None}

algorithms = ["SCAFFOLD", "FedAvg"]
noises = [True, False]
similarities = [1, 0.1, 0]
# algorithms = ["SCAFFOLD"]
# noises = [True]
# similarities = [0]

for similarity in similarities:
    print("Downloading dataset")
    generate_femnist_data(similarity, 100, 20)
    for noise in noises:
        for algorithm in algorithms:
            input_dict["algorithm"] = algorithm
            input_dict["similarity"] = similarity
            input_dict["noise"] = noise
            simulate(**input_dict)


plot_by_similarities(input_dict["dataset"], algorithms, noises, similarities, input_dict["num_glob_iters"])

# plot_dict = get_plot_dict(input_dict, algorithms, epochs)
# plot_by_epochs(**plot_dict)
