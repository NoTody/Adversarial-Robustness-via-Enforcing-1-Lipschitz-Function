import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, proxy, proxy_optim, gamma, num_iter=1):
        super(AdvWeightPerturb, self).__init__()
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.num_iter = num_iter

    def calc_awp(self, model, inputs_adv, targets):
        self.proxy.load_state_dict(model.state_dict())
        self.proxy.train()
        for i in range(self.num_iter):
            loss = - F.cross_entropy(self.proxy(inputs_adv), targets)
            print(f"loss: {loss}")
            self.proxy_optim.zero_grad()
            loss.backward()
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(model, self.proxy)
        return diff

    def perturb(self, model, diff):
        add_into_weights(model, diff, coeff=1.0 * self.gamma)

    def restore(self, model, diff):
        add_into_weights(model, diff, coeff=-1.0 * self.gamma)

