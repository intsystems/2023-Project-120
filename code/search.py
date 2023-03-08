# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN
from utils import accuracy, MyDartsTrainer


logger = logging.getLogger('nni')
dataset = "fashionmnist"

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--decay", default=0.0, type=float, help='regularization coefficient')
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--save_path", default='finar_architecture.json', type=str)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset(dataset)

    if dataset == "fashionmnist":
        model = CNN(32, 1, args.channels, 10, args.layers)
    if dataset == "cifar10":
        model = CNN(32, 3, args.channels, 10, args.layers)

    criterion = nn.CrossEntropyLoss() # mycriterion()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    trainer = MyDartsTrainer( # MyDartsTrainer
        model=model,
        loss=criterion, # =mycriterion,
        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
        optimizer=optim,
        num_epochs=args.epochs,
        dataset=dataset_train,
        batch_size=args.batch_size,
        log_frequency=args.log_frequency,
        unrolled=args.unrolled,
        tau=0.95, # параметр сглаживания для подсчета дивергенции
        decay=args.decay # вес регуляризации
    )
    trainer.fit()
    final_architecture = trainer.export()
    print('Final architecture:', trainer.export())
    json.dump(trainer.export(), open(args.save_path, 'w'))
