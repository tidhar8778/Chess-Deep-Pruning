import torch.nn as nn
# If you change the model architecture, you should train a new model using nn.py and only then you can use it to play a game
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout):
        hidden_size1, hidden_size2, hidden_size3, hidden_size4 = hidden_sizes
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        out = self.relu(out)
        out = self.fc5(out)
        return out
