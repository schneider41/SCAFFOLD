import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MclrLogistic(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MclrLogistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class MclrCrossEntropy(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MclrCrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs


class DNN(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class LinearRegression(nn.Module):
    def __init__(self, input_dim=60, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs


class CifarNet(nn.Module):
    def __init__(self, categories=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  # doubled bias learning rate
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)  # doubled bias learning rate
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)  # doubled bias learning rate
        # self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, categories)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.MaxPool2d(3, 2)(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.AvgPool2d(3, 2)(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = nn.AvgPool2d(3, 2)(x)
        x = torch.flatten(x, 1)
        # x = x.view(-1, 4800)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output