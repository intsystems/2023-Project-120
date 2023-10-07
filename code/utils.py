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

from nni.retiarii.oneshot.pytorch import DartsTrainer
import torch.nn.functional as F
import torch
import json
from torch.distributions import RelaxedOneHotCategorical


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
                 weight=1e3, lambd=0, tau=0.9, t=0.2, optimalPath='checkpoints/fashionMNIST/optimal/arc.json', train_as_optimal=False):
        super().__init__(model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip,
                 learning_rate, batch_size, workers,
                 device, log_frequency,
                 arc_learning_rate, unrolled)
        self.weight = weight
        self.lambd = lambd
        self.tau = tau
        self.t = t
        self.train_as_optimal = train_as_optimal
        
        operations = { "maxpool": 0, "avgpool": 1, "skipconnect": 2, "sepconv3x3": 3,
                      "sepconv5x5": 4, "dilconv3x3" : 5, "dilconv5x5" : 6 } # индексы операций по названиям (в соответствии с nas_modules)
        O = len(operations) # кол-во операций

        self.optimal = {} # выдает сглаженный тензор операций по названию операции
        with open(optimalPath) as f:
            checkpoint_optimum = json.load(f) # оптимальная архитектура в виде словаря

        for name, _ in self.nas_modules:
            if name in checkpoint_optimum.keys() and name[-6:] != 'switch': # если ребро есть в оптимальной архитектуре и модуль не reduce_n#_switch
                operation = checkpoint_optimum[name] # имя оптимальной операции

                index = operations[operation] # индекс оптимальной операции
                t = torch.zeros(O, device=('cuda' if torch.cuda.is_available() else 'cpu'))
                t[index] = 1

                t = t * self.tau + 1 / O * (1 - self.tau)

                self.optimal[name] = t

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
    
    def edgeComparison(self):
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
    

    def _logits_and_loss(self, X, y):
        logits = self.model(X)
        if self.train_as_optimal:
            loss = self.loss(logits, y)
        else:    
            loss = self.loss(logits, y) + (self.lambd - self.edgeComparison()) ** 2
        # self.decay * self.JSD() # обращаем внимание, что регуляризатор не влияет на первый уровень оптимизации
        return logits, loss
