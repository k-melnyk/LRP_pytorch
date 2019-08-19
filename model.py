import os
from os.path import join as jpath
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import SGD

import numpy as np

import matplotlib.pyplot as plt

import lrp


def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output


class ReshapeForLinear(nn.Module):
    def __init__(self, size):
        self.size = size
        super(ReshapeForLinear, self).__init__()

    def forward(self, x):
        return x.view(-1, self.size[0] * self.size[1] * self.size[2])

    def relprop(self, R):
        return R.view(-1, self.size[0], self.size[1], self.size[2])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layers = nn.Sequential(
            lrp.Conv2d(1, 16, kernel_size=3, padding=1),
            lrp.BatchNorm2d(16),
            lrp.ReLU(),
            lrp.MaxPool2d(2),

            lrp.Conv2d(16, 32, kernel_size=5, padding=2),
            lrp.BatchNorm2d(32),
            lrp.ReLU(),
            lrp.MaxPool2d(2),

            ReshapeForLinear([32, 7, 7]),

            lrp.Linear(in_features=7*7*32, out_features=1024),
            lrp.ReLU(),
            lrp.Linear(in_features=1024, out_features=512),
            lrp.ReLU(),
            lrp.Linear(in_features=512, out_features=10)
        )

    def forward(self, x, y=None):
        out = self.layers(x)

        if y is not None:
            criterion = nn.CrossEntropyLoss()
            y = autograd.Variable(y)
            loss = criterion(out, y)
            return out, loss
        else:
            return out

    def relprop(self, R):
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l-1].relprop(R)
        return R


def train_model(train_loader, test_loader, device, lr, epochs, output_path, valid_loader=False):
    model = CNN().to(device)
    optimizer = SGD(model.parameters(), lr=lr)

    average_loss_train = []
    average_loss_test = []

    accuracy_train = []
    accuracy_test = []

    for epoch in range(epochs):
        model.train()
        correct_train, loss_train, _ = loop_dataset(model, train_loader, device, optimizer)

        print(f'Epoch {epoch} : average train loss - {np.mean(loss_train)}, train accuracy - {correct_train}')

        average_loss_train.append(np.mean(loss_train))
        accuracy_train.append(correct_train)

        model.eval()
        correct_test, loss_test, _ = loop_dataset(model, test_loader, device)

        print(f'Epoch {epoch} : average test loss - {np.mean(loss_test)}, test accuracy - {correct_test}')

        average_loss_test.append(np.mean(loss_test))
        accuracy_test.append(correct_test)

    model.eval()

    for i in range(0, len(model.layers)):
        model.layers[i].register_forward_hook(forward_hook)
    if valid_loader:
        correct_valid, _, output = loop_dataset(model, valid_loader, device)

        print('\033[99m'+f'Accuracy on VALID test: {correct_valid}'+'\033[0m')

    checkpoint = {'model': CNN(),
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(output_path, 'checkpoint.pth'))

    plt.figure()
    plt.plot(range(epochs), average_loss_train, lw=0.3, c='g')
    plt.plot(range(epochs), average_loss_test, lw=0.3, c='r')
    plt.legend(['train loss', 'test_loss'])
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig(jpath(output_path, 'loss.png'))

    plt.figure()
    plt.plot(range(epochs), accuracy_train, lw=0.3, c='g')
    plt.plot(range(epochs), accuracy_test, lw=0.3, c='r')
    plt.legend(['train_acc', 'test_acc'])
    plt.xlabel('#Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(jpath(output_path, 'accuracy.png'))


def loop_dataset(model, loader, device, optimizer=None):
    losses = []
    total = 0
    correct = 0
    predicted = []
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        out, loss = model(inputs, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, pred = torch.max(out.data, 1)
        predicted.append(out)
        total += labels.size(0)
        correct += (pred == labels).sum()
        losses.append(loss.data.cpu().detach().numpy())

        pred = out.data.max(1, keepdim=True)[1]

        pred = pred.squeeze().cpu().numpy()
        pred = (pred[:, np.newaxis] == np.arange(10)) * 1.0
        pred = torch.from_numpy(pred).type(torch.FloatTensor)
        pred = torch.autograd.Variable(pred).to(device)

        lrp = model.relprop(out.data * pred)
        a = lrp[0].cpu().detach().numpy()
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 1))
        lrp = a / np.max(np.abs(a))

        plt.figure()
        plt.imshow(inputs[0].cpu().detach().numpy().squeeze(), cmap='gray')

        plt.figure()
        plt.imshow(lrp, cmap='seismic', clim=(-1, 1))

    return float(correct) / float(total), losses, predicted