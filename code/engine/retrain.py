# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets as datasets
import utils as utils
from model import CNN
from nni.retiarii.oneshot.pytorch.utils import AverageMeter
from nni.retiarii import fixed_arch
# from nni.nas import fixed_arch
# from nni.nas import model_context

logger = logging.getLogger('nni')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

def train(train_loader, model, optimizer, criterion, epoch, args):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args['GRAD_CLIP'])
        optimizer.step()

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % args['LOG_FREQUENCY'] == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, args['EPOCHS'], step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, args['EPOCHS'], top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step, args):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc5"], bs)

            if step % args['LOG_FREQUENCY'] == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, args['EPOCHS'], step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, args['EPOCHS'], top1.avg))

    return top1.avg

def retrain(arc_path, args, dataset_train, dataset_valid):
    print(f'Start retraining architecture in {arc_path}...')
    with fixed_arch(arc_path):
        model = utils.get_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args['LEARNING_RATE'], momentum=0.9, weight_decay=args['WEIGHT_DECAY'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['EPOCHS'], eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args['BATCH_SIZE'],
                                            shuffle=True,
                                            num_workers=args['WORKERS'],
                                            pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                            batch_size=args['BATCH_SIZE'],
                                            shuffle=False,
                                            num_workers=args['WORKERS'],
                                            pin_memory=True)

    best_top1 = 0.
    for epoch in range(args['EPOCHS']):
        drop_prob = args['DROP_PATH_PROB'] * epoch / args['EPOCHS']
        model.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch, args)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step, args)
        best_top1 = max(best_top1, top1)

        lr_scheduler.step()

    arc_path_parts = arc_path.split('/')
    mod_name = 'mod' + arc_path_parts[-1][3:]
    mod_path_parts = arc_path_parts[:-1] + [mod_name]
    mod_path = '/'.join(mod_path_parts)
    torch.save(model.state_dict(), mod_path)
    # torch.save(model.state_dict(), args.save_folder + "/mod.json")
    print("Final best Prec@1 = {:.4%}".format(best_top1))
    print()
    return { mod_path : best_top1 }
    

def collect_all(workbench):
    arc_paths = set()
    for root, _, files in os.walk(workbench):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.split('/')[-1][:3] == 'arc':
                arc_paths.add(file_path)
    return arc_paths

def collect_all_folder(folder):
    arcs_found = set()
    for file in os.listdir(folder):
        if file[:3] == 'arc':
            arcs_found.add(folder + '/' + file)
    return arcs_found

if __name__ == '__main__':
    args = utils.get_config('configs/retrain.yaml')

    print('Loading dataset...')
    dataset_train, dataset_valid = datasets.get_dataset(args['DATASET'])
    print()

    workbench = utils.get_save_path(args)

    if args['ALL'] == True:
        arcs_for_retrain = collect_all(workbench)
    else:
        for dir in args['DIRS']:
            arcs_for_retrain = set()
            if dir == 'random':
                for common_edges in args['COMMON_EDGES']:
                    folder = workbench + f'/random/amount={common_edges}'
                    arcs_for_retrain.update(collect_all_folder(folder))
            if dir == 'optimal':
                folder = workbench + f'/optimal'
                arcs_for_retrain.update(collect_all_folder(folder))
            if dir == 'hypernet':
                for number in args['HYPERNET_NUMBERS']:
                    # print('ASDASDASDAD', workbench + f'/hypernet/{number}')
                    for lam_dir in os.listdir(workbench + f'/hypernet/{number}'):
                        arc_path = workbench + f'/hypernet/{number}/' + lam_dir + '/arc.json'
                        if os.path.isfile(arc_path):
                            arcs_for_retrain.add(arc_path)
            if dir == 'edges':
                for lambdas in args['LAMBDAS']:
                    folder = workbench + f'/random/amount={common_edges}'
                    arcs_for_retrain.update(collect_all_folder(folder))

    if not args['FORCE_RETRAIN']:
        arcs_already_retrained = set()
        for arc_path in arcs_for_retrain:
            arc_path_parts = arc_path.split('/')
            mod_name = 'mod' + arc_path_parts[-1][3:]
            mod_path_parts = arc_path_parts[:-1] + [mod_name]
            mod_path = '/'.join(mod_path_parts)
            if os.path.isfile(mod_path):
                arcs_already_retrained.add(arc_path)
        arcs_for_retrain = arcs_for_retrain - arcs_already_retrained

    print('Arcs that will be retrained:')
    for arc_path in sorted(list(arcs_for_retrain)):
        print(arc_path)
    print()

    res_dict = {}
    for arc_path in sorted(list(arcs_for_retrain)):
        res_dict.update(retrain(arc_path, args, dataset_train, dataset_valid))
    
    for k, v in res_dict.items():
        print(f'Accuracy of model in {k} = {v}')

