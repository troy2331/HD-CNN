import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super(CrossEntropyLoss, self).__init__()
        self.args = args

    def forward(self, pred, target):
        loss = nn.CrossEntropyLoss()
        return loss(pred, target)


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args

    def forward(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)


class SmoothL1Loss(nn.Module):
    def __init__(self, args):
        super(SmoothL1Loss, self).__init__()
        self.args = args

    def forward(self, pred, target):
        loss = nn.SmoothL1Loss()
        return loss(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, args, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, args, weight=None, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = args.smoothing
        self.confidence = 1.0 - self.smoothing
        self.weight = weight
        self.cls = args.class_n
        self.dim = dim
        self.adjacency = args.adjacency

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            if self.adjacency == False:
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(
                    1, target.data.unsqueeze(1), self.confidence)
            if self.adjacency == True:
                for idx, target in enumerate(target.data.unsqueeze(1)):
                    fill_value = self.smoothing / 2
                    if int(target) == 0:
                        true_dist[idx][int(target)+1] = fill_value * 1.5
                        true_dist[idx][int(target)] += fill_value * 0.5
                    elif int(target) == (self.cls-1):
                        true_dist[idx][int(target)-1] = fill_value * 1.5
                        true_dist[idx][int(target)] += fill_value * 0.5
                    else:
                        true_dist[idx][int(target)-1] = fill_value
                        true_dist[idx][int(target)+1] = fill_value
                    true_dist[idx][int(target)] += self.confidence
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


_criterion_entropoints = {
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "l1": L1Loss,
    "smooth_l1": SmoothL1Loss
}


def criterion_entrypoint(criterion_name):
    return _criterion_entropoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entropoints


def create_criterion(args):
    criterion_name = args.criterion
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(args)
    else:
        raise RuntimeError("Unkwon criterion (%s)" % criterion_name)
    return criterion
