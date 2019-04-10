import sys
import os
from argparse import ArgumentParser
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

from dataloader import read_bci_data

def choose_act_func(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    else:
        raise TypeError('activation_function type not defined.')


class EEGNet(nn.Module):
    def __init__(self, act_func):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential()
        # [B, 1, 2, 750] -> [B, 16, 2, 750]
        self.firstConv.add_module('conv1', nn.Conv2d(1, 16, (1, 51), 1, (0, 25), bias=False))
        self.firstConv.add_module('norm1', nn.BatchNorm2d(16))

        self.depthwiseConv = nn.Sequential()
        # [B, 16, 2, 750] -> [B, 32, 1, 750]
        self.depthwiseConv.add_module('conv2', nn.Conv2d(16, 32, (2, 1), groups=16, bias=False))
        self.depthwiseConv.add_module('norm2', nn.BatchNorm2d(32))
        self.depthwiseConv.add_module('act1', choose_act_func(act_func))
        # [B, 32, 1, 750] -> [B, 32, 1, 187]
        self.depthwiseConv.add_module('pool1', nn.AvgPool2d((1, 4), stride=(1, 4)))
        self.depthwiseConv.add_module('drop1', nn.Dropout(p=0.25))
        
        self.separableConv = nn.Sequential()
        self.separableConv.add_module('conv3', nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False))
        self.separableConv.add_module('norm3', nn.BatchNorm2d(32))
        self.separableConv.add_module('act2', choose_act_func(act_func))
        # [B, 32, 1, 187] -> [B, 32, 1, 23]
        self.separableConv.add_module('pool2', nn.AvgPool2d((1, 8), stride=(1, 8)))
        self.separableConv.add_module('drop2', nn.Dropout(p=0.25))

        self.classify = nn.Sequential(
            nn.Linear(32*23, 2)
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        res = x.view(x.size(0), -1)     # [B, 32, 1, 23] -> [B, 32 * 23]
        out = self.classify(res)
        return out


class DeepConvNet(nn.Module):
    def __init__(self, act_func):
        super(DeepConvNet, self).__init__()
        self.block1 = nn.Sequential()
        # [B, 1, 2, 750] -> [B, 25, 2, 750]
        self.block1.add_module('conv1', nn.Conv2d(1, 25, (1, 5), bias=False))
        # [B, 25, 2, 750] -> [B, 25, 1, 750]
        self.block1.add_module('conv2', nn.Conv2d(25, 25, (2, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25))
        self.block1.add_module('act1', choose_act_func(act_func))
        # [B, 25, 2, 750] -> [B, 25, 1, 373]
        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p=0.5))

        self.block2 = nn.Sequential()
        # [B, 25, 1, 373] -> [B, 50, 1, 373]
        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 5), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50))
        self.block2.add_module('act2', choose_act_func(act_func))
        # [B, 50, 1, 373] -> [B, 50, 1, 184]
        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p=0.5))
        
        self.block3 = nn.Sequential()
        # [B, 50, 1, 184] -> [B, 100, 1, 184]
        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 5), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100))
        self.block3.add_module('act3', choose_act_func(act_func))
        # [B, 100, 1, 184] -> [B, 100, 1, 90]
        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p=0.5))

        self.block4 = nn.Sequential()
        # [B, 100, 1, 90] -> [B, 200, 1, 90]
        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 5), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200))
        self.block4.add_module('act4', choose_act_func(act_func))
        # [B, 200, 1, 90] -> [B, 200, 1, 43]
        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p=0.5))

        self.classify = nn.Sequential(
            nn.Linear(200*43, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        res = x.view(x.size(0), -1)     # [B, 200, 1, 43] -> [B, 200 * 43]
        out = self.classify(res)
        return out


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def show_result(x, y_dict, model_name=''):
    fig, ax = plt.subplots()

    for key,val in y_dict.items():
        ax.plot(x, val, label=key)

    ax.set_title("Actvation function comparision ({})".format(model_name))
    plt.show()


def cal_accuracy(net, x, y):
    t_x = torch.from_numpy(x.astype(np.float32))
    gt = torch.from_numpy(y.astype(np.float32))
    pred_y = net(t_x)
    pred_y = torch.max(pred_y, 1)[1].data.numpy()
    accuracy = float((pred_y == gt.data.numpy()).astype(int).sum()) / float(gt.size(0))
    return accuracy


def handle_param(args, net):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    else:
        raise TypeError('optimizer type not defined.')
    if args.loss_function == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    else:
        raise TypeError('loss_function type not defined.')
    return optimizer, loss_function


def choose_net(args):
    if args.model == 'eeg':
        return {
        'elu': [EEGNet('elu')],
        'relu': [EEGNet('relu')],
        'lrelu': [EEGNet('lrelu')],
        }
    elif args.model == 'dcn':
        return {
        'elu': [DeepConvNet('elu')],
        'relu': [DeepConvNet('relu')],
        'lrelu': [DeepConvNet('lrelu')],
        }
    else:
        raise TypeError('model type not defined.')


def main(args):
    train_x, train_y, test_x, test_y = read_bci_data()
    torch_dataset = Data.TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch, shuffle=True)
    
    net_dict = choose_net(args)
    model_fullname = {'eeg': 'EEGNet', 'dcn': 'DeepConvNet'}
    acc_dict = {}
    if args.load:
        net_dict['relu'][0].load_state_dict(torch.load(args.load))
        net_dict['relu'][0].eval()
        test_accuracy = cal_accuracy(net_dict['relu'][0], test_x, test_y)
        print('test_acc: {:.4f}%'.format(test_accuracy * 100))
        return
    # net[0]: model, net[1]: optimizer, net[2]: loss_function
    for key,net in net_dict.items():
        acc_dict['train_{}'.format(key)] = []
        acc_dict['test_{}'.format(key)] = []
        optimizer, loss_func = handle_param(args, net[0])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150], gamma=0.5)
        net.extend([optimizer, loss_func, scheduler])
    max_acc = 0
    file_name = '{}_lr{}_ep{}'.format(args.model, args.learning_rate, args.epochs)
    # start training
    for epoch in range(args.epochs):
        print('-'*10, 'epoch', epoch+1, '-'*10)
        loss_dict = {}
        for key in net_dict.keys():
            loss_dict[key] = []
        # training
        for b_x, b_y in train_loader:
            for key,net in net_dict.items():
                # apply scheduler
                net[3].step()
                output = net[0](b_x)
                loss = net[2](output, b_y.long())
                loss_dict[key].append(loss.data.numpy())
                net[1].zero_grad()
                loss.backward()
                net[1].step()
        # show loss and accuracy
        for key,net in net_dict.items():
            net[0].eval()
            train_accuracy = cal_accuracy(net[0], train_x, train_y)
            test_accuracy = cal_accuracy(net[0], test_x, test_y)
            if test_accuracy > max_acc:
                max_acc = test_accuracy
                torch.save(net[0].state_dict(), file_name + '.pkl')
            acc_dict['train_{}'.format(key)].append(train_accuracy)
            acc_dict['test_{}'.format(key)].append(test_accuracy)
            print('---------- {} ({}) ----------'.format(model_fullname[args.model], key))
            print('training loss: {:.6f} | train_acc: {:.6f} | test_acc: {:.6f}'.format(max(loss_dict[key]), train_accuracy, test_accuracy))
            net[0].train()
    print('max_acc: {}'.format(max_acc))
    
    # save / show result
    # show_result(range(args.epochs), acc_dict, 'EEG')
    with open(file_name + '.json', 'w') as f:
        json.dump({
            'x': list(range(args.epochs)),
            'y_dict': acc_dict,
            'title': model_fullname[args.model],
        }, f, cls=NumpyEncoder)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=64)
    parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=1e-2)
    parser.add_argument("-ep", "--epochs", help="your training target", type=int, default=150)
    parser.add_argument("-opt", "--optimizer", help="adam | rmsp", type=str, default='adam')
    parser.add_argument("-lf", "--loss-function", help="loss function", type=str, default='CrossEntropy')
    # parser.add_argument("-act", "--activation-function", help="elu | relu | lrelu", type=str, default='elu')
    parser.add_argument("-m", "--model", help="eeg | dcn", type=str, default='eeg')
    parser.add_argument("-load", "--load", help="your pkl file path", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()
