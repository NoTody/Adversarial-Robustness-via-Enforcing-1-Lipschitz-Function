import numpy as np
import torch

import advertorch.attacks as attacks

class EvalBN(object):
    def __init__(self, adversary):
        self.adversary = adversary
        self.model = self.adversary.predict

    def __call__(self, *args, **kwargs):
        mode = self.model.training
        self.model.eval()
        return_ = self.adversary(*args, **kwargs)
        self.model.train(mode)
        return return_


def attack_module(model, criterion, _eval=False):
    #if P.distance == 'Linf':
    _ORD = np.inf
    _PGD_ALPHA = 2. / 255.
    _PGD_EPS = 8. / 255.
    _PGD_ITER = 10
    # elif P.distance == 'L2':
    #     _ORD = 2
    #     _PGD_ALPHA = 15. / 255.
    #     _PGD_EPS = 128. / 255.
    #     _PGD_ITER = 10
    # elif P.distance == 'L1':
    #     _ORD = 1
    #     _PGD_ALPHA = 400. / 255.
    #     _PGD_EPS = 2000. / 255.
    #     _PGD_ITER = 10

    #if epsilon is None:
    epsilon = _PGD_EPS
    #if alpha is None:
    alpha = _PGD_ALPHA
    #if P.n_iters is None:
    n_iters = _PGD_ITER

    adv_kwargs = {'loss_fn': criterion, 'clip_min': 0, 'clip_max': 1}
    # if P.adv_method == 'fgsm':
    #     adv_kwargs.update({'eps': epsilon})
    #     if P.distance == 'Linf':
    #         adversary = attacks.GradientSignAttack(model, **adv_kwargs)
    #     elif P.distance == 'L2':
    #         adversary = attacks.GradientAttack(model, **adv_kwargs)
    #     else:
    #         raise NotImplementedError()
    #elif P.adv_method == 'pgd':
    adv_kwargs.update({'eps': epsilon, 'eps_iter': alpha, 'ord': _ORD,
                        'nb_iter': n_iters, 'rand_init': True})
    adversary = attacks.PGDAttack(model, **adv_kwargs)

    return adversary
    