# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from model import CNN
from nni.retiarii.oneshot.pytorch.utils import AverageMeter
from nni.retiarii import fixed_arch
from glob import glob

logger = logging.getLogger('nni')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.1, type=float)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--checkpoints-folder", default="./checkpoints")

    args = parser.parse_args()
    dataset_train, dataset_valid = datasets.get_dataset("fashionmnist", cutout_length=16)

    models = []
    for dir in glob(args.checkpoints_folder + "/*"):
        print(dir)
        with fixed_arch(dir + "/arc.json"):
            model = CNN(32, 1, 36, 10, args.layers, auxiliary=True)
        model.eval()
        model.to(device)
        model.load_state_dict(torch.load(dir + "/mod.json"))
        
        models.append(model)

    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    # validation
    softmax = nn.Softmax(dim=1)
    for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            probabilities = softmax(models[0](X))
            for i in range(1, len(models)):
                 probabilities += softmax(models[i](X))
            probabilities = probabilities / len(models)
            loss = criterion(probabilities, y)

            accuracy = utils.accuracy(probabilities, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc5"], bs)

            if step % 10 == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    logger.info("Final best Prec@1 = {:.4%}".format(top1.avg))
