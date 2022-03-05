import torch
import torch.nn as nn
import torch.nn.functional as F

import math
fl = math.floor

class LeNet(nn.Module):
    def __init__(self, image_size, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.last_map_x = fl((fl((image_size[1]-4)/2)-4)/2)
        self.last_map_y = fl((fl((image_size[2]-4)/2)-4)/2)

        self.linear1 = nn.Linear(16 * self.last_map_x * self.last_map_y, 120)
        self.linear2 = nn.Linear(120, 84)
        # self.out_layer = nn.Linear(84, num_classes)
        self.out_layer = nn.Identity()

    def forward(self, inp):
        x = self.pool1(F.relu(self.conv1(inp)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.last_map_x * self.last_map_y)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        otp = self.out_layer(x)
        return otp


class CNNRNN(nn.Module):
    def __init__(self, out_size):
        super(CNNRNN, self).__init__()
        self.rnn_in_size = 84
        self.hidden_size = 100

        self.cnn = LeNet((1, 28, 28), 10)
        self.rnn = nn.LSTMCell(self.rnn_in_size, self.hidden_size)

        self.out_layer = nn.Linear(self.hidden_size, out_size)

    def forward(self, inp, hid_in):
        x = self.cnn(inp)
        hid_out = self.rnn(x, hid_in)
        otp = self.out_layer(hid_out[0])
        return otp, hid_out

    def init_hidden(self, batch_size):
        init_hid = (torch.zeros(batch_size, self.hidden_size),
                    torch.zeros(batch_size, self.hidden_size))
        return init_hid


class LeNetAttention(nn.Module):
    def __init__(self, image_size, num_classes=10):
        super(LeNetAttention, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.last_map_x = fl((fl((image_size[1]-4)/2)-4)/2)
        self.last_map_y = fl((fl((image_size[2]-4)/2)-4)/2)

        self.linear1 = nn.Linear(16 * self.last_map_x * self.last_map_y, 120)
        self.linear2 = nn.Linear(120, 84)
        # self.out_layer = nn.Linear(84, num_classes)
        self.out_layer = nn.Identity()

    def forward(self, inp, attention):
        beta, gamma = attention
        x = self.pool1(F.relu(self.conv1(inp)))
        x = gamma[:, :, None, None] * x + beta[:, :, None, None]
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.last_map_x * self.last_map_y)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        otp = self.out_layer(x)
        return otp


class CNNRNNFeedback(nn.Module):
    def __init__(self, out_size):
        super(CNNRNNFeedback, self).__init__()
        self.rnn_in_size = 84
        self.hidden_size = 100

        self.cnn = LeNetAttention((1, 28, 28), 10)
        self.rnn = nn.LSTMCell(self.rnn_in_size, self.hidden_size)

        self.out_layer = nn.Linear(self.hidden_size, out_size)
        self.cnn_attention_beta_layer = nn.Linear(self.hidden_size, 1)
        self.cnn_attention_gamma_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, inp, hid_in):
        rnn_hid_in, attention = hid_in

        x = self.cnn(inp, attention)
        hid_out = self.rnn(x, rnn_hid_in)

        beta = self.cnn_attention_beta_layer(rnn_hid_in[0])
        gamma = self.cnn_attention_gamma_layer(rnn_hid_in[0])
        attention = (beta, gamma)
        otp = self.out_layer(rnn_hid_in[0])

        return otp, (hid_out, attention)

    def init_hidden(self, batch_size):
        init_hid = (torch.zeros(batch_size, self.hidden_size),
                    torch.zeros(batch_size, self.hidden_size))
        init_attention = (torch.zeros(batch_size, 1), torch.zeros(batch_size, 1))
        return init_hid, init_attention