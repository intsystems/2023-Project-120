# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn

import ops as ops
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
                
def p(P_MIN, P_MAX):
    return torch.rand(1).item() * (P_MAX - P_MIN) + P_MIN


from nni.retiarii.oneshot.pytorch import DartsTrainer
import torch.nn.functional as F
import torch
import json
from torch.distributions import RelaxedOneHotCategorical
import logging
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device
_logger = logging.getLogger(__name__)
import utils as utils
import numpy as np

import torch.nn.init as init

# class PWNet(nn.Module):
#     def __init__(self, size, kernel_num,  init_ = 'random'):    
#         nn.Module.__init__(self)
        
#         if not isinstance(size, tuple): # check if size is 1d
#             size = (size,)
            
#         self.size = size
#         self.kernel_num = kernel_num
               
#         total_size = [kernel_num] + list(self.size) # [kenrel_num x param_size]

#         self.const = nn.Parameter(torch.randn(total_size, dtype=torch.float32))
#         if init_ == 'random':
#             for i in range(kernel_num):
#                 if len(self.size) > 1:
#                     init.kaiming_uniform_(self.const.data[i], a=np.sqrt(5))
#                 else:
                
#                     self.const.data[i]*=0
#                     self.const.data[i]+=torch.randn(size)
#         else:
#             self.const.data *=0
#             self.const.data += init_
        
#         self.pivots = nn.Parameter(torch.tensor(np.linspace(P_MIN + 0.1, P_MAX - 0.1, kernel_num - 2)), requires_grad=True)
        
            
#     def forward(self, lam):           
#         # lam_ = lam * 0.99999
#         if lam < self.pivots[0]:
#             dist = (self.pivots[0] - lam) / (self.pivots[0] - P_MIN)
#             res = self.const[0] * (dist) + (1.0 - dist) * self.const[1]
#         elif lam > self.pivots[-1]:
#             dist = (P_MAX - lam) / (P_MAX - self.pivots[-1])
#             res = self.const[-2] * (dist) + (1.0 - dist) * self.const[-1]
#         else:
#             right = 0
#             while self.pivots[right] < lam:
#                 right += 1
#             left = right - 1

#             dist = (self.pivots[right] - lam) / (self.pivots[right] - self.pivots[left])
#             res = self.const[left] * (dist) + (1.0 - dist) * self.const[right]
#         return res

# class Linear(nn.Module):
#     def __init__(self, nas_modules):    
#         nn.Module.__init__(self)
        
#         self.names = []

#         consts = []

#         for name, module in nas_modules:
#             self.names.append(name)
#             size = (module.alpha.size()[0], )
#             total_size = [2] + list(size)
#             new_param = torch.randn(total_size)
#             consts.append(new_param)
#         self.consts = nn.ParameterList(consts)
            
#             # self.net_modules.append((name, new_param))
            
#     def forward(self, lam):
#         ret = {}
#         for name, const in zip(self.names, self.consts):
#             ret.update({name : const[0] * lam + const[1]})
#         return ret

class Alpha(nn.Module):
    def __init__(self, nas_modules, device):    
        nn.Module.__init__(self)
        self.names = []
        
        consts = []

        for name, module in nas_modules:
            self.names.append(name)
            size = (module.alpha.size()[0])
            new_param = torch.randn(size, device=device)
            consts.append(new_param)
        self.consts = nn.ParameterList(consts)

    def forward(self):
        ret = {}
        for name, const in zip(self.names, self.consts):
            ret.update({name : const})
        return ret

class PWLinear(nn.Module):
    def __init__(self, nas_modules, N, device, P_MIN, P_MAX):    
        nn.Module.__init__(self)
        self.nas_modules = nas_modules
        self.N = N
        self.device = device
        self.P_MIN = P_MIN
        self.P_MAX = P_MAX

        self.pivots = torch.linspace(self.P_MIN + 1 / N, self.P_MAX - 1 / N, N - 2)

        self.alphas = nn.ModuleList([Alpha(nas_modules, device)])  # 0
        for _ in self.pivots:
            self.alphas.append(Alpha(nas_modules, device))
        self.alphas.append(Alpha(nas_modules, device)) # N - 1

        self.pivots = nn.Parameter(self.pivots)

            
    def forward(self, lam):
        self.pivots

        right_index = torch.searchsorted(self.pivots, lam, side='right')
        left_index = right_index - 1

        ret = {}
        left_pivot = (self.P_MIN if left_index <= -1 else self.pivots[left_index])
        left = self.alphas[left_index + 1]() # left_params
        right_pivot = (self.P_MAX if right_index >= self.N - 2 else self.pivots[right_index])
        right = self.alphas[right_index + 1]() # right_params

        assert right_pivot >= lam
        assert left_pivot <= lam
        # print(left_index, right_index, left)
        for name, _ in self.nas_modules:
            k_left = torch.tensor((lam - left_pivot) / (right_pivot - left_pivot), device=self.device)
            k_right = torch.tensor(1 - (lam - left_pivot) / (right_pivot - left_pivot), device=self.device)
            ret.update({ name : k_left * left[name] + k_right.to(self.device) * right[name] })
        return ret

# def JSD(net_1_logits, net_2_logits):
#     from torch.functional import F
#     net_1_probs = F.softmax(net_1_logits, dim=0)
#     net_2_probs = F.softmax(net_2_logits, dim=0)
    
#     total_m = 0.5 * (net_1_probs + net_2_probs)
    
#     loss = 0.0
#     loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
#     loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
#     return (0.5 * loss)

class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]

class MyDartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(MyDartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = torch.randn(len(self.op_choices)) * 1e-3

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(MyDartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]


class MyDartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(MyDartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = torch.randn(input_choice.n_candidates) * 1e-3
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(MyDartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]

class EdgeNES(DartsTrainer):
    def __init__(self, model, metrics,
                 num_epochs, regime, dataset, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False,
                 weight_start=None, weight_end=None,
                 t_start=None, t_end=None,
                 lambd=None,
                 p_min=None, p_max=None,
                 kernel_num=None,
                 tau=None,
                 optimal_arc_dict=None,
                 n_chosen=2,
                 ):
        assert regime in ['optimal', 'hypernet', 'edges'], 'Regime of model shold be optimal or hypernet or edges'

        self.model = model
        self.regime = regime
        self.loss = nn.CrossEntropyLoss()
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency
        self.model.to(self.device)
        self.unrolled = unrolled
        self.grad_clip = grad_clip
        self.n_chosen = n_chosen

        # General setup
        self._init_dataloader()
        optim = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=3.0E-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs, eta_min=0.001)
        self.model_optim = optim
        
        # NAS modules
        if self.regime in ['edges', 'optimal']:
            self.nas_modules = []
            replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
            replace_input_choice(self.model, DartsInputChoice, self.nas_modules)
            for _, module in self.nas_modules:
                module.to(self.device)
        elif self.regime == 'hypernet':
            self.nas_modules = []
            replace_layer_choice(self.model, MyDartsLayerChoice, self.nas_modules)
            replace_input_choice(self.model, MyDartsInputChoice, self.nas_modules)
            for _, module in self.nas_modules:
                module.to(self.device)

        # Arcitecture optimizer
        if self.regime in ['edges', 'optimal']:
            ctrl_params = {}
            for _, m in self.nas_modules:
                if m.name in ctrl_params:
                    assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                    m.alpha = ctrl_params[m.name]
                else:
                    ctrl_params[m.name] = m.alpha
            self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), arc_learning_rate, betas=(0.5, 0.999),
                                            weight_decay=1.0E-3)
        elif self.regime == 'hypernet':
            # hypernetwork params
            self.p_min = p_min
            self.p_max = p_max
            self.kernel_num = kernel_num

            self.init_network()
            self.ctrl_optim = torch.optim.Adam(self.hypernetwork.parameters(), arc_learning_rate, betas=(0.5, 0.999),
                                    weight_decay=1e-3)

        # Regularizer params
        if self.regime in ['edges', 'hypernet']:
            self.weight_start, self.weight_end = weight_start, weight_end
            self.t_start, self.t_end = t_start, t_end
            self.tau = tau
            self.optimal_arc_dict = optimal_arc_dict
            self.set_numerical_optimal_arc()

        if self.regime == 'edges':
            self.lambd = lambd

    def set_numerical_optimal_arc(self):
        operations = { "maxpool": 0, "avgpool": 1, "skipconnect": 2, "sepconv3x3": 3,
                      "sepconv5x5": 4, "dilconv3x3" : 5, "dilconv5x5" : 6 } # индексы операций по названиям (в соответствии с nas_modules)
        O = len(operations)

        self.optimal_arc_numerical = {} # архитектура в виде векторов, где 1 стоят там, где есть ребро
        for name, _ in self.nas_modules:
            if name[-6:] != 'switch': # если модуль не reduce_n#_switch
                operation = self.optimal_arc_dict[name] # имя оптимальной операции
                index = operations[operation] # индекс оптимальной операции
                t = torch.zeros(O, device=self.device)
                t[index] = 1
                t = t * self.tau + 1 / O * (1 - self.tau)
                self.optimal_arc_numerical[name] = t
            elif name[-6:] == 'switch':
                parents = self.optimal_arc_dict[name]
                node = int(name[-8])
                t = torch.zeros(node)
                t[parents] = 1
                self.optimal_arc_numerical[node] = t # 1 стоят там, где ребро есть, 0 там, где ребра нет

    def init_network(self):
        self.hypernetwork = PWLinear(self.nas_modules, self.kernel_num, self.device, self.p_min, self.p_max)
        self.hypernetwork.to(self.device)

    # def JSD(self):
    #     '''
    #     Подсчет дивергенции между своей и оптимальной архитектурой
    #     '''
    #     res = 0.0
    #     count = 0
    #     for name, module in self.nas_modules: # суммируем диаергенцию по всем ребрам
    #         if name in self.optimal.keys():
    #             res += JSD(module.alpha, torch.log(self.optimal[name]))
    #             count += 1
    #     return res / count

    # def edgeComparisonOldVersion(self):
    #     '''
    #     Регуляризатор на основе количество общих  ребер
    #     '''
    #     count = 0
    #     sum = 0
    #     for name, module in self.nas_modules: # суммируем диаергенцию по всем ребрам
    #         if name in self.optimal.keys():
    #             print(F.softmax(module.alpha, dim=0), type(F.softmax(module.alpha, dim=0)))
    #             alpha = F.softmax(module.alpha, dim=0)
    #             alpha0 = RelaxedOneHotCategorical(probs=self.optimal[name], temperature=self.t).rsample().t()
    #             print(torch.dot(alpha, alpha0))
    #             sum += torch.dot(alpha, alpha0)
    #             count += 1
    #     return sum / count
    
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
                alpha_opt = self.optimal_arc_numerical[name]
                # alpha0 = RelaxedOneHotCategorical(probs=self.optimal[name], temperature=self.t).rsample().t()
                p, n = int(name[-1]), int(name[-4]) # номер parent и node
                sum += torch.dot(alpha, alpha_opt) * beta[n][p] * self.optimal_arc_numerical[n][p]
        return sum

    def _logits_and_loss(self, X, y, lambd=None):
        logits = self.model(X)
        if self.regime == 'optimal' or lambd is None:
            loss = self.loss(logits, y)
        else:
            loss = self.loss(logits, y) + self.weight * (lambd - self.edgeCount()) ** 2
        return logits, loss

    def common_edges_with_opt(self):
        optimal_arc = self.optimal_arc_dict
        arc = self.export()
        return utils.common_edges(arc, optimal_arc)

    def set_alpha(self, architecture):
        for name, module in self.nas_modules:
            module.alpha = architecture[name]

    def get_arch(self, lam):
        architecture = self.hypernetwork(lam)
        self.set_alpha(architecture)
        return self.export()

    def _train_one_epoch(self, epoch, writer=None):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            if self.regime == 'hypernet':
                lambd = p(self.p_min, self.p_max)
            elif self.regime == 'edges':
                lambd = self.lambd
            elif self.regime == 'optimal':
                lambd = None

            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)

            # set architecture from hypernet
            if self.regime == 'hypernet':
                architecture = self.hypernetwork(lambd)
                self.set_alpha(architecture)
            
            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                _, loss = self._logits_and_loss(val_X, val_y, lambd)
                loss.backward()
            self.ctrl_optim.step()

            # set architecture from hypernet
            if self.regime == 'hypernet':
                architecture = self.hypernetwork(lambd)
                self.set_alpha(architecture)

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
                # print(lambd, next(self.hypernetwork.parameters()))
                print(f'Epoch [{epoch + 1}/{self.num_epochs}] Step [{step + 1}/{len(self.train_loader)}]  {meters}')
                if writer is not None:
                    writer.add('loss', epoch * len(self.train_loader) + step, loss.item())
                    # writer.add('edges', epoch * len(self.train_loader) + step, self.common_edges_with_opt())
                    writer.add('accuracy', epoch * len(self.train_loader) + step, meters['acc1'].val)
        return meters
    
    def warmup_weight(self, epoch, epochs):
        return self.weight_start+ (epoch / epochs) * (self.weight_end - self.weight_start)

    def warmup_t(self, epoch, epochs):
        return self.t_start + (epoch / epochs) * (self.t_end - self.t_start)
                
    def fit(self):
        for i in range(self.num_epochs):
            if self.regime in ['edges', 'hypernet']:
                self.weight = self.warmup_weight(i, self.num_epochs)
                self.t_alpha = self.warmup_t(i, self.num_epochs)
                self.t_beta = self.warmup_t(i, self.num_epochs)
            # if writer is not None:
            #     writer.add('weight', i, self.weight)
            #     writer.add('tempreture', i, self.t_beta)
            self._train_one_epoch(i)
