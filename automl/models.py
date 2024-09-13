import torch.nn as nn

class BaselineNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class EvolvableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, evolved_activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.evolved_activation = evolved_activation

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.evolved_activation(x)
        x = self.fc2(x[0])
        return x