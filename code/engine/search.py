from importlib import reload
import json
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt

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
    print('Reading optimal architecture from ' + save_path + '...')
    with open(save_path) as f:
        optimal_arc_dict = json.load(f)
    print('Optimal arcitecture:', optimal_arc_dict)
    print()
    return optimal_arc_dict

def get_random_operation():
    operations = [ "maxpool", "avgpool", "skipconnect", "sepconv3x3", "sepconv5x5", "dilconv3x3", "dilconv5x5"]
    return np.random.choice(operations)

def make_dirs(args):
    print('Making dirs...')
    path = utils.get_save_path(args)
    utils.make_dir(args['SAVE_FOLDER'])
    utils.make_dir(path)
    utils.make_dir(path + '/optimal')
    utils.make_dir(path + '/edges')
    utils.make_dir(path + '/random')
    utils.make_dir(path + '/hypernet')
    if args['REGIME'] == 'edges':
        for lambd in args['LAMBDAS']:
            utils.make_dir(path + f'/edges/lam={lambd}')
    if args['REGIME'] == 'random':
        for lambd in args['COMMON_EDGES']:
            utils.make_dir(path + f'/random/amount={lambd}')
    print()

if __name__ == "__main__":
    args = utils.get_config('configs/search.yaml')

    print('Loading dataset...')
    dataset_train, dataset_valid = datasets.get_dataset(args['DATASET'])
    print()

    path = utils.get_save_path(args) + '/' + args['REGIME']

    make_dirs(args)

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
                n_chosen=args['N_CHOSEN'],
            )
            trainer.fit()
            final_architecture = trainer.export()
            print('Final architecture:', final_architecture)
            number = max([-1] + [int(file[4:file.find('.json')]) for file in os.listdir(path)]) + 1
            save_path = path + f'/arc_{number}.json'
            print('Saving to ' + save_path + '...')
            utils.save_arc(final_architecture, save_path)
            print()
            

    if args['REGIME'] == 'random':
        optimal_arc_dict = get_optimal_arc(args)
        swithes = []
        for lambd in args['COMMON_EDGES']:
            new_arc_dict = optimal_arc_dict.copy()
            print(f"Start randoming architecture for common edges = {lambd}...")
            edges_to_change = np.random.choice(4, 4 - lambd, replace=False)
            for edge in edges_to_change:
                node = edge + 2
                new_parent = np.random.randint(0, node)
                while new_parent in optimal_arc_dict[f'reduce_n{node}_switch']:
                    new_parent = np.random.randint(0, node)
                new_arc_dict[f'reduce_n{node}_switch'] = [new_parent]
                new_arc_dict[f"reduce_n{node}_p{new_parent}"] = get_random_operation()
            print('Final architecture:', new_arc_dict)
            number = max([-1] + [int(file[4:file.find('.json')]) for file in os.listdir(path + f'/amount={lambd}')]) + 1
            save_path = path + f'/amount={lambd}/arc_{number}.json'
            print('Saving to ', save_path + '...')
            utils.save_arc(new_arc_dict, save_path)
            # print(utils.common_edges(optimal_arc_dict, new_arc_dict))
            print()
        
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
                weight_start=args['WEIGHT_START'],
                weight_end=args['WEIGHT_END'],
                t_start=args['T_START'],
                t_end=args['T_END'],
                lambd=lambd,
            )
            trainer.fit()
            final_architecture = trainer.export()
            print('Final architecture:', final_architecture)
            number = max([-1] + [int(file[4:file.find('.json')]) for file in os.listdir(path + f'/lam={lambd}')]) + 1
            save_path = path + f'/lam={lambd}/arc_{number}.json'
            print('Saving to ', save_path, '...')
            utils.save_arc(final_architecture, save_path)
            print()

    if args['REGIME'] == 'hypernet':
        number = max([-1] + [int(file) for file in os.listdir(path)]) + 1
        utils.make_dir(path + f'/{number}')
        for lambd in args['HYPERNET_LAMBDAS']:
            utils.make_dir(path + f'/{number}/lam={lambd}')

        optimal_arc_dict = get_optimal_arc(args)
    
        print(f"Start training hypernet number {number}...")
        model = utils.get_model(args)

        trainer = EdgeNES(
            model=model,
            metrics=lambda output, target: utils.accuracy(output, target, topk=(1,)),
            num_epochs=args['EPOCHS'],
            dataset=dataset_train,
            batch_size=args['BATCH_SIZE'],
            log_frequency=args['LOG_FREQUENCY'],
            unrolled=args['UNROLLED'],
            regime='hypernet',
            learning_rate=args['LEARNING_RATE'],
            arc_learning_rate=args['ARC_LEARNING_RATE'],
            n_chosen=args['N_CHOSEN'],
            optimal_arc_dict=optimal_arc_dict,
            tau=args['TAU'],
            weight_start=args['WEIGHT_START'],
            weight_end=args['WEIGHT_END'],
            t_start=args['T_START'],
            t_end=args['T_END'],
            p_min=args['P_MIN'],
            p_max=args['P_MAX'],
            kernel_num=args['KERNEL_NUM'],
        )
        trainer.fit()
        for lambd in args['HYPERNET_LAMBDAS']:
            sampled_architecture = trainer.get_arch(lambd)
            print(f'Architecture sampled for lambda = {lambd}:', sampled_architecture)
            # number = max([-1] + [int(file[4:file.find('.json')]) for file in os.listdir(path + f'/lam={lambd}')]) + 1
            save_path = path + f'/{number}/lam={lambd}/arc.json'
            print('Saving to ', save_path, '...')
            utils.save_arc(sampled_architecture, save_path)
            print()



