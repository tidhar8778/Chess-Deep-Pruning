import torch.nn as nn
import torch
from params import num_classes, dropout, option_nn
# If you change the model architecture, you should train a new model using nn.py and only then you can use it to play a game
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # input_vector = [17, 244, 128, 1]

        self.fc1_1 = nn.Linear(17, 15)
        self.fc1_2 = nn.Linear(244, 32)
        self.fc1_3 = nn.Linear(128, 32)

        self.fc2 = nn.Linear(80, 32)
        self.fc3 = nn.Linear(32, 1) # 1 for sigmoid, 2 for cross entropy

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # second option
        self.fco_1 = nn.Linear(390, 128)
        self.fco_2 = nn.Linear(128, 32)
        self.fco_3 = nn.Linear(32, 2)

        # third option
        self.fc31 = nn.Linear(769, 128)
        self.fc32 = nn.Linear(128, 32)
        self.fc33 = nn.Linear(32, 2)

    def forward(self, x):
        # input_vector = [17, 244, 128, 1]
        if option_nn == 1:
            activation = self.relu
            out1_1 = self.dropout(activation(self.fc1_1(x[:, :17])))
            out1_2 = self.dropout(activation(self.fc1_2(x[:, 17:17+244])))
            out1_3 = self.dropout(activation(self.fc1_3(x[:, 17+244:17+244+128])))
            out1_4 = x[:, 17+244+128:]

            out2 = self.dropout(activation(self.fc2(torch.cat((out1_1, out1_2, out1_3, out1_4), -1))))
            out = self.fc3(out2)

            return out

        elif option_nn == 2:
        # option 2
            activation = self.relu

            out = self.fco_1(x)
            out = activation(out)
            out = self.dropout(out)

            out = self.fco_2(out)
            out = activation(out)
            out = self.dropout(out)

            out = self.fco_3(out)


            return out

        elif option_nn == 3:
            activation = self.relu

            out = self.dropout(activation(self.fc31(x)))
            out = self.dropout(activation(self.fc32(out)))
            out = self.fc33(out)

            return out
