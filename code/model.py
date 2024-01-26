# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn

import ops
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect, n_chosen):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                LayerChoice(OrderedDict([
                    ("maxpool", ops.PoolBN('max', channels, 3, stride, 1, affine=False)),
                    ("avgpool", ops.PoolBN('avg', channels, 3, stride, 1, affine=False)),
                    ("skipconnect", nn.Identity() if stride == 1 else ops.FactorizedReduce(channels, channels, affine=False)),
                    ("sepconv3x3", ops.SepConv(channels, channels, 3, stride, 1, affine=False)),
                    ("sepconv5x5", ops.SepConv(channels, channels, 5, stride, 2, affine=False)),
                    ("dilconv3x3", ops.DilConv(channels, channels, 3, stride, 2, 2, affine=False)),
                    ("dilconv5x5", ops.DilConv(channels, channels, 5, stride, 4, 2, affine=False))
                ]), label=choice_keys[-1]))
        self.drop_path = ops.DropPath()
        self.input_switch = InputChoice(n_candidates=len(choice_keys), n_chosen=n_chosen, label="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)


class Cell(nn.Module):

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction, n_chosen):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0, n_chosen=n_chosen))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output


class CNN(nn.Module):
    
    def __init__(self, input_size, in_channels, channels, n_classes, n_layers, n_nodes=4,
                 stem_multiplier=3, auxiliary=False, n_chosen=2):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction, n_chosen)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

            if i == self.aux_pos:
                self.aux_head = AuxiliaryHead(input_size // 4, channels_p, n_classes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)

        if aux_logits is not None:
            return logits, aux_logits
        return logits

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================

from nni.retiarii.oneshot.pytorch import DartsTrainer
import torch.nn.functional as F
import torch
import json
from torch.distributions import RelaxedOneHotCategorical
import logging
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup
_logger = logging.getLogger(__name__)
import utils

def JSD(net_1_logits, net_2_logits):
    from torch.functional import F
    net_1_probs = F.softmax(net_1_logits, dim=0)
    net_2_probs = F.softmax(net_2_logits, dim=0)
    
    total_m = 0.5 * (net_1_probs + net_2_probs)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * loss)


class MyDartsTrainer(DartsTrainer):
    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False,
                 weight=1e3, 
                 lambd=0, 
                 tau=0.9, 
                 t_alpha=0.3, t_beta=0.4,
                 optimalPath='checkpoints/fashionMNIST/optimal/arc.json', 
                 train_as_optimal=False,
                 n_chosen=2,
                 ):
        super().__init__(model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip,
                 learning_rate, batch_size, workers,
                 device, log_frequency,
                 arc_learning_rate, unrolled)
        self.weight = weight
        self.lambd = lambd
        self.tau = tau
        self.t_alpha = t_alpha
        self.t_beta = t_beta
        self.train_as_optimal = train_as_optimal
        self.n_chosen = n_chosen

        if train_as_optimal:
            return
        
        operations = { "maxpool": 0, "avgpool": 1, "skipconnect": 2, "sepconv3x3": 3,
                      "sepconv5x5": 4, "dilconv3x3" : 5, "dilconv5x5" : 6 } # индексы операций по названиям (в соответствии с nas_modules)
        O = len(operations) # кол-во операций

        self.optimal_arc = {} # архитектура в виде векторов, где 1 стоят там, где есть ребро

        with open(optimalPath) as f:
            self.checkpoint_optimum = json.load(f) # оптимальная архитектура в виде словаря

        for name, _ in self.nas_modules:
            if name[-6:] != 'switch': # если ребро есть в оптимальной архитектуре и модуль не reduce_n#_switch
                operation = self.checkpoint_optimum[name] # имя оптимальной операции
                index = operations[operation] # индекс оптимальной операции
                t = torch.zeros(O, device=self.device)
                t[index] = 1
                t = t * self.tau + 1 / O * (1 - self.tau)
                self.optimal_arc[name] = t
            elif name[-6:] == 'switch':
                parents = self.checkpoint_optimum[name]
                n = int(name[-8])
                t = torch.zeros(n)
                t[parents] = 1
                self.optimal_arc[n] = t # 1 стоят там, где ребро есть, 0 там, где ребра нет


    def JSD(self):
        '''
        Подсчет дивергенции между своей и оптимальной архитектурой
        '''
        res = 0.0
        count = 0
        for name, module in self.nas_modules: # суммируем диаергенцию по всем ребрам
            if name in self.optimal.keys():
                res += JSD(module.alpha, torch.log(self.optimal[name]))
                count += 1
        return res / count

    def edgeComparisonOldVersion(self):
        '''
        Регуляризатор на основе количество общих  ребер
        '''
        count = 0
        sum = 0
        print(self.nas_modules)
        for name, module in self.nas_modules: # суммируем диаергенцию по всем ребрам
            if name in self.optimal.keys():
                print(F.softmax(module.alpha, dim=0), type(F.softmax(module.alpha, dim=0)))
                alpha = F.softmax(module.alpha, dim=0)
                alpha0 = RelaxedOneHotCategorical(probs=self.optimal[name], temperature=self.t).rsample().t()
                print(torch.dot(alpha, alpha0))
                sum += torch.dot(alpha, alpha0)
                count += 1
        return sum / count
    
    def edgeCount(self):
        '''
        Регуляризатор на основе количество общих  ребер
        '''
        sum = 0
        beta = {}
        for name, module in self.nas_modules:
            if name[-6:] == 'switch':
                n = int(name[-8]) # номер ноды
                beta[n] = RelaxedOneHotCategorical(logits=module.alpha, temperature=self.t_beta).rsample().t() # записываем распределения по ребрам

        for name, module in self.nas_modules: # разность по ребрам
            if name[-6:] != 'switch':
                alpha = RelaxedOneHotCategorical(logits=module.alpha, temperature=self.t_alpha).rsample().t()
                alpha_opt = self.optimal_arc[name]
                # alpha0 = RelaxedOneHotCategorical(probs=self.optimal[name], temperature=self.t).rsample().t()
                p, n = int(name[-1]), int(name[-4]) # номер parent и node
                sum += torch.dot(alpha, alpha_opt) * beta[n][p] * self.optimal_arc[n][p]
        return sum
    
    def get_nas_modeles(self):
        return self.nas_modules

    def _logits_and_loss(self, X, y):
        logits = self.model(X)
        if self.train_as_optimal:
            loss = self.loss(logits, y)
        else:
            loss = self.loss(logits, y) + self.weight * (self.lambd - self.edgeCount()) ** 2
        # self.decay * self.JSD() # обращаем внимание, что регуляризатор не влияет на первый уровень оптимизации
        return logits, loss

    def common_edges_with_opt(self):
        optimal_arc = self.checkpoint_optimum
        arc = self.export()
        return utils.common_edges(arc, optimal_arc)

    def _train_one_epoch(self, epoch, writer):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                self._backward(val_X, val_y)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            metrics = self.metrics(logits, trn_y)
            metrics['loss'] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}] Step [{step + 1}/{len(self.train_loader)}]  {meters}')
                if writer is not None:
                    writer.add('loss', epoch * len(self.train_loader) + step, loss.item())
                    writer.add('edges', epoch * len(self.train_loader) + step, self.common_edges_with_opt())
                    writer.add('accuracy', epoch * len(self.train_loader) + step, meters['acc1'].val)
        return meters
                
    def fit(self, writer=None, warmup_weight=None, warmup_t=None):
        for i in range(self.num_epochs):
            if warmup_weight is not None:
                self.weight = warmup_weight(i, self.num_epochs)
            if warmup_t is not None:
                self.t_alpha = warmup_t(i, self.num_epochs)
                self.t_beta = warmup_t(i, self.num_epochs)
            if writer is not None:
                writer.add('weight', i, self.weight)
                writer.add('tempreture', i, self.t_beta)
            self._train_one_epoch(i, writer)
                


