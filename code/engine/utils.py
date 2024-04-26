def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

#==========================================================================
import matplotlib.pyplot as plt
from model import CNN
import os
import json
import yaml

def make_dir(new_folder_path):
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

def get_model(args):
    if args['DATASET'] == 'fashionmnist':
        model = CNN(32, 1, args['CHANNELS'], 10, args['LAYERS'])
    if args['DATASET'] == 'cifar10':
        model = CNN(32, 3, args['CHANNELS'], 10, args['LAYERS'])
    if args['DATASET'] == 'cifar100':
        model = CNN(32, 3, args['CHANNELS'], 100, args['LAYERS'])
    return model

def get_save_path(args):
    return args['SAVE_FOLDER'] + '/' + args['DATASET']

def warmup_weight(epoch, epochs, MIN=0.1, MAX=2):
    return MIN + (epoch / epochs) * (MAX - MIN)

def warmup_t(epoch, epochs, MIN=0.01, MAX=0.5):
    return MIN + (epoch / epochs) * (MAX - MIN)

def save_arc(arcitecture, file_path):
    with open(file_path, "w+") as file:
        json.dump(arcitecture, file, indent=4)

def get_config(path):
    print(f'Reading {path}...')
    with open(path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    print('Configuration for searching:')
    for key, value in args.items():
        print(f'{key} : {value}')
    print()
    return args

class Writer:
    def __init__(self):
        self.data = {}
    def new_name(self, name):
        self.data.update({ name : [[], []]})
    def add(self, name, x, y):
        self.data[name][0].append(x)
        self.data[name][1].append(y)
    def get(self, name):
        return self.data[name][0], self.data[name][1]
    def clean_name(self, name):
        self.data[name][0], self.data[name][1] = [], []
    def clean_all(self):
        for name in self.data.keys():
            self.clean_name(name)
    def show(self):
        N = len(self.data.keys())
        fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(18, 4))
        for i, name in enumerate(self.data.keys()):
            ax[i].set_title(name)
            if name in ['edges', 'loss', 'accuracy']:
                ax[i].set_xlabel('iterations')
            else:
                ax[i].set_xlabel('epochs')
            x, y = self.get(name)
            ax[i].plot(x, y)

def common_edges(left, right):
    same = 0
    for n in range(2, 6):
        common_parents = set(left[f"reduce_n{n}_switch"]) & set(right[f"reduce_n{n}_switch"])
        for p in common_parents:
            key = f"reduce_n{n}_p{p}"
            if left[key] == right[key]:
                same += 1
    return same
