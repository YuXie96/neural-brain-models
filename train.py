import torch
import torch.nn as nn
import torch.nn.functional as F


import os.path as osp

import torchvision
from config_global import ROOT_DIR
from torch.utils.data import DataLoader
from models import CNNRNNFeedback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import math
fl = math.floor


class DelayedMatch:
    def __init__(self, sample_step, delay_step, test_step, batch_size):
        self.sample_step = sample_step
        self.delay_step = delay_step
        self.test_step = test_step

        # comparison loss
        self.criterion = nn.BCEWithLogitsLoss()

        assert batch_size % 2 == 0, 'batch size must be odd number'
        self.batch_size = batch_size
        self.split_size = int(batch_size / 2)

    def roll(self, model, data_batch):
        input_, label_ = data_batch

        # assuming the same image is not sampled twice in the batch
        inp1 = input_[:self.split_size]
        inp2 = input_[self.split_size:]

        # first self.split_size trial in the batch are match
        # last self.split_size trial in the batch are non-match
        sample_input = torch.cat((inp1, inp1), 0)
        match_input = torch.cat((inp1, inp2), 0)

        target = torch.zeros((self.batch_size, 1))
        target[:self.split_size, 0] = 1.0

        roll_step = self.sample_step + self.delay_step + self.test_step

        task_loss = 0
        pred_num = 0
        pred_correct = 0
        hidden = model.init_hidden(self.batch_size)
        for t_ in range(roll_step):
            if t_ < self.sample_step:
                model_inp = sample_input
            elif self.sample_step <= t_ < self.sample_step + self.delay_step:
                model_inp = torch.zeros_like(sample_input)
            else:
                model_inp = match_input

            output, hidden = model(model_inp, hidden)

            if t_ >= self.sample_step + self.delay_step:
                task_loss += self.criterion(output, target)

                pred_num += target.size(0)
                pred_tf = output > 0.0
                pred_correct += (pred_tf == target).sum().item()

        task_loss = task_loss / self.test_step
        return task_loss, pred_num, pred_correct


if __name__ == '__main__':
    b_size = 20
    num_wks = 2

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    data_set = torchvision.datasets.MNIST(root=osp.join(ROOT_DIR, 'data'),
                                          train=True, download=True,
                                          transform=trans)
    data_loader = DataLoader(data_set, batch_size=b_size, shuffle=True,
                             num_workers=num_wks, drop_last=True)


    model = CNNRNNFeedback(1)
    optimizer = torch.optim.Adam(model.parameters())
    task = DelayedMatch(5, 0, 5, b_size)

    batch_number = 0
    for data in data_loader:
        batch_number += 1
        if batch_number >= 1000:
            break

        loss, pred_num, pred_correct = task.roll(model, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_number % 100 == 0:
            print('Loss: {}'.format(loss.item()))
            print('Accuracy: {}%'.format(100 * pred_correct / pred_num))
