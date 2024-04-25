from importlib import reload
import json
import logging
import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import datasets
from model import EdgeNES

import utils
import yaml
import os
import numpy as np

logger = logging.getLogger('nni')


def get_optimal_arc(args):
    path = utils.get_save_path(args)
    save_path = path + '/' + 'optimal/' + 'arc_' + str(args['OPTIMAL_NUMBER']) + '.json'
    print('Loading optimal architecture from ' + save_path + '...')
    with open(save_path) as f:
        optimal_arc_dict = json.load(f)
    print('Optimal arcitecture:', optimal_arc_dict)
    print()
    return optimal_arc_dict

def get_config():
    print('Reading configs/search.yaml...')
    with open("configs/search.yaml", "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    print('Configuration for searching:')
    for key, value in args.items():
        print(f'{key} : {value}')
    print()
    return args

if __name__ == "__main__":
    args = get_config()

    print('Loading dataset...')
    dataset_train, dataset_valid = datasets.get_dataset(args['DATASET'])
    print()

    path = utils.get_save_path(args)

    print('Making dirs...')
    utils.make_dir(path + '/optimal')
    utils.make_dir(path + '/edges')
    utils.make_dir(path + '/random')
    if args['REGIME'] == 'edges':
        for lambd in args['LAMBDAS']:
            utils.make_dir(path + f'/edges/lam={lambd}')
    if args['REGIME'] == 'random':
        for lambd in args['COMMON_EDGES']:
            utils.make_dir(path + f'/random/amount={lambd}')
    print()

    if args['REGIME'] == 'optimal':
        for i in range(args['OPTIMAL_AMOUNT']):
            print(f"Start searching optimal architecture [{i+1}/{args['OPTIMAL_AMOUNT']}]...")
            model = utils.get_model(args)
            trainer = EdgeNES(
                model=model,
                metrics=lambda output, target: utils.accuracy(output, target, topk=(1,)),
                num_epochs=args['EPOCHS'],
                dataset=dataset_train,
                batch_size=args['BATCH_SIZE'],
                log_frequency=args['LOG_FREQUENCY'],
                unrolled=args['UNROLLED'],
                regime='optimal',
                learning_rate=args['LEARNING_RATE'],
                arc_learning_rate=args['ARC_LEARNING_RATE'],
                n_chosen=args['N_CHOSEN']
            )
            trainer.fit()
            final_architecture = trainer.export()
            print('Final architecture:', final_architecture)
            number = max([-1] + [int(file[4:file.find('.json')]) for file in os.listdir(path + '/optimal')]) + 1
            print('Saving to ', path + f'/optimal/arc_{number}.json...')
            json.dump(final_architecture, open(path + f'/optimal/arc_{number}.json', 'w+'))
            print()
            

    if args['REGIME'] == 'random':
        optimal_arc_dict = get_optimal_arc(args)
        for lambd in args['COMMON_EDGES']:
            edges_to_change = np.random.choice(8, 4 - lambd, replace=False)
            for edge in edges_to_change:
                pass


    if args['REGIME'] == 'edges':
        optimal_arc_dict = get_optimal_arc(args)
        for lambd in args['LAMBDAS']:
            print(f"Start searching architecture for lambda = {lambd}...")
            model = utils.get_model(args)

            trainer = EdgeNES(
                model=model,
                metrics=lambda output, target: utils.accuracy(output, target, topk=(1,)),
                num_epochs=args['EPOCHS'],
                dataset=dataset_train,
                batch_size=args['BATCH_SIZE'],
                log_frequency=args['LOG_FREQUENCY'],
                unrolled=args['UNROLLED'],
                regime='edges',
                learning_rate=args['LEARNING_RATE'],
                arc_learning_rate=args['ARC_LEARNING_RATE'],
                n_chosen=args['N_CHOSEN'],
                optimal_arc_dict=optimal_arc_dict,
                tau=args['TAU'],
                weight_func=utils.warmup_weight,
                t_func=utils.warmup_t,
                lambd=lambd,
            )
            trainer.fit()
            final_architecture = trainer.export()
            print('Final architecture:', final_architecture)
            number = max([-1] + [int(file[4:file.find('.json')]) for file in os.listdir(path + f'/edges/lam={lambd}')]) + 1
            save_path = path + f'/edges/lam={lambd}/arc_{number}.json'
            print('Saving to ', save_path, '...')
            utils.save_arc(final_architecture, save_path)
            print()




