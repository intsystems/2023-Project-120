# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets as datasets
import utils as utils
from nni.retiarii.oneshot.pytorch.utils import AverageMeter
from nni.retiarii import fixed_arch

logger = logging.getLogger('nni')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

def inference(models, valid_loader):
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
    return top1.avg

if __name__ == "__main__":
    args = utils.get_config('configs/inference.yaml')

    utils.fix_seed(args['SEED'])

    print('Loading dataset...')
    dataset_train, dataset_valid = datasets.get_dataset(args['DATASET'])
    print()

    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                                batch_size=args['BATCH_SIZE'],
                                                shuffle=False,
                                                num_workers=args['WORKERS'],
                                                pin_memory=True)

    workbench = utils.get_save_path(args)

    path_ensembles = {}
    
    arc_model_path_list = []
    if args['DIR'] == 'random':
        number = args['NUMBER_RANDOM']
        for common_edges in args['COMMON_EDGES']:
            folder = workbench + f'/random/amount={common_edges}'
            arc_path = folder + f'/arc_{number}.json'
            mod_path = folder + f'/mod_{number}.json'
            arc_model_path_list.append((arc_path, mod_path))

    if args['DIR'] == 'optimal':
        folder = workbench + f'/optimal'
        for number in args['OPTIMAL_NUMBERS']:
            arc_path = folder + f'/arc_{number}.json'
            mod_path = folder + f'/mod_{number}.json'
            arc_model_path_list.append((arc_path, mod_path))

        # for epoch in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
        #     arc_model_path_list = []
        #     for number in [10, 15, 17]: # [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
        #         arc_path = folder + f'/arc_{number}_e{epoch}.json'
        #         mod_path = folder + f'/mod_{number}_e{epoch}.json'
        #         arc_model_path_list.append((arc_path, mod_path))
        #     path_ensembles.update({ f'e{epoch}' : arc_model_path_list })
    
    if args['DIR'] == 'hypernet':
        number = args['HYPERNET_NUM']
        folder = workbench + f'/hypernet/{number}'
        for lam in args['HYPERNET_LAMBDAS']:
            arc_path = folder + f'/lam={lam}/arc.json'
            mod_path = folder + f'/lam={lam}/mod.json'
            arc_model_path_list.append((arc_path, mod_path))

        # for epoch in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        #     arc_model_path_list = []
        #     for lam in [2, 3, 4]:
        #         arc_path = folder + f'/lam={lam}/arc_e{epoch}.json'
        #         mod_path = folder + f'/lam={lam}/mod_e{epoch}.json'
        #         arc_model_path_list.append((arc_path, mod_path))
        #     path_ensembles.update({ f'e{epoch}' : arc_model_path_list })
    
    if args['DIR'] == 'edges':
        number = args['NUMBER_EDGES']
        for lambd in args['EDGES_LAMBDAS']:
            arc_path = workbench + f'/edges/lam={lambd}/arc_{number}.json'
            mod_path = workbench + f'/edges/lam={lambd}/mod_{number}.json'
            arc_model_path_list.append((arc_path, mod_path))

    # ===========================================================================
    print('Architectures included into ensemble locations:')
    for arc, mod in arc_model_path_list:
        print(arc)
    print()

    models = []
    for arc, mod in arc_model_path_list:
            with fixed_arch(arc):
                model = utils.get_model(args)
            model.eval()
            model.to(device)
            model.load_state_dict(torch.load(mod))
            
            models.append(model)
    

    inference(models, valid_loader)
    # ===========================================================================

    # ===========================================================================
    # for key, val in path_ensembles.items():
    #     print(f'Ensemble of {key}:')
    #     for v in val:
    #         print(v[1])
    # print()

    # ensembles_results = {}
    # for key, val in path_ensembles.items():
    #     models = []
    #     for arc, mod in val:
    #         with fixed_arch(arc):
    #             model = utils.get_model(args)
    #         model.eval()
    #         model.to(device)
    #         model.load_state_dict(torch.load(mod))
    #         models.append(model)

    #     top_1_acc = inference(models, valid_loader)

    #     ensembles_results.update({ key : top_1_acc })
    
    # for key, val in ensembles_results.items():
    #     print(f'Result of ensembling from {key} is {val}')
    # ===========================================================================