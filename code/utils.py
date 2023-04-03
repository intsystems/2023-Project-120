# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

import json

import numpy as np
import torch
from nni.retiarii.oneshot.pytorch import DartsTrainer
from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice


def JSD(net_1_logits, net_2_logits):
    from torch.functional import F
    net_1_probs = F.softmax(net_1_logits, dim=0)
    net_2_probs = F.softmax(net_2_logits, dim=0)

    total_m = 0.5 * (net_1_probs + net_2_probs)

    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean")
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean")
    return (0.5 * loss)


def get_optimal_probabilities(nas_modules, operations, tau):
    O = len(operations) # кол-во операций

    optimal = {} # выдает сглаженный тензор операций по названию операции
    with open('checkpoints/0/arc.json') as f:
        checkpoint_optimum = json.load(f) # оптимальная архитектура в виде словаря

    for name, _ in nas_modules:
        if name in checkpoint_optimum.keys() and name[-6:] != 'switch': # если ребро есть в оптимальной архитектуре и модуль не reduce_n#_switch
            operation = checkpoint_optimum[name] # имя оптимальной операции

            index = operations[operation] # индекс оптимальной операции
            t = torch.zeros(O, device=('cuda' if torch.cuda.is_available() else 'cpu'))
            t[index] = 1

            t = t * tau + 1 / O * (1 - tau)

            optimal[name] = t

    return optimal


class MyDartsTrainer(DartsTrainer):
    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False, tau=0.95, decay=0):
        super().__init__(model, loss, metrics, optimizer,
                         num_epochs, dataset, grad_clip,
                         learning_rate, batch_size, workers,
                         device, log_frequency,
                         arc_learning_rate, unrolled)
        self.tau = tau
        self.decay = decay

        operations = { "maxpool": 0, "avgpool": 1, "skipconnect": 2, "sepconv3x3": 3,
                       "sepconv5x5": 4, "dilconv3x3" : 5, "dilconv5x5" : 6 } # индексы операций по названиям (в соответствии с nas_modules)

        self.optimal = get_optimal_probabilities(self.nas_modules, operations, tau)

    def JSD(self): # подсчет дивергенции между своей и оптимальной архитектурой
        res = 0.0
        count = 0
        for name, module in self.nas_modules: # суммируем диаергенцию по всем ребрам
            if name in self.optimal.keys():
                res += JSD(module.alpha, torch.log(self.optimal[name]))
                count += 1
        return res / count

    def _logits_and_loss(self, X, y):
        logits = self.model(X)
        loss = self.loss(logits, y) - self.decay * self.JSD() # обращаем внимание, что регуляризатор не влияет на первый уровень оптимизации
        return logits, loss


class BayesianSearchMC(DartsTrainer):
    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False, tau=0.95, seed=0):
        super().__init__(model, loss, metrics, optimizer,
                         num_epochs, dataset, grad_clip,
                         learning_rate, batch_size, workers,
                         device, log_frequency,
                         arc_learning_rate, unrolled)

        np.random.seed(seed)

        operations = { "maxpool": 0, "avgpool": 1, "skipconnect": 2, "sepconv3x3": 3,
                       "sepconv5x5": 4, "dilconv3x3" : 5, "dilconv5x5" : 6 }
        self.O = len(operations)
        self.tau = tau

        self.optimal = get_optimal_probabilities(self.nas_modules, operations, tau)

    def fit(self):
        pass

    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                if isinstance(module, DartsLayerChoice):
                    idx = np.random.choice(self.O, p=np.array(self.optimal[name].cpu()))
                    result[name] = list(module.op_choices.keys())[idx]
                else:
                    result[name] = module.export()
        return result
