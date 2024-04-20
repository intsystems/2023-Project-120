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
import matplotlib.pyplot as plt

class Writer:
    def __init__(self):
        self.data = {}
    def new_name(self, name):
        self.data.update({ name : [[], []]})
    def add(self, name, x, y):
        self.data[name][0].append(x)
        self.data[name][1].append(y)
    def get(self, name):
        return self.data[name][0], self.data[name][1]
    def clean_name(self, name):
        self.data[name][0], self.data[name][1] = [], []
    def clean_all(self):
        for name in self.data.keys():
            self.clean_name(name)
    def show(self):
        N = len(self.data.keys())
        fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(18, 4))
        for i, name in enumerate(self.data.keys()):
            ax[i].set_title(name)
            if name in ['edges', 'loss', 'accuracy']:
                ax[i].set_xlabel('iterations')
            else:
                ax[i].set_xlabel('epochs')
            x, y = self.get(name)
            ax[i].plot(x, y)

def common_edges(left, right):
    same = 0
    for n in range(2, 6):
        common_parents = set(left[f"reduce_n{n}_switch"]) & set(right[f"reduce_n{n}_switch"])
        for p in common_parents:
            key = f"reduce_n{n}_p{p}"
            if left[key] == right[key]:
                same += 1
    return same
