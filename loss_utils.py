import torch
import torch.nn.functional as F

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5

def _H_min_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)
 
    logsoftmax_prob1 = torch.log(prob1.clamp(min=1e-8))
    logsoftmax_prob2 = torch.log(prob2.clamp(min=1e-8))
    logsoftmax_mean = torch.log(mean_prob.clamp(min=1e-8))

    min_d = logsoftmax_mean - torch.min(logsoftmax_prob1, logsoftmax_prob2) 
    return min_d.mean()
