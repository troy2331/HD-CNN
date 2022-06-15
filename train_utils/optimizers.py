import torch.optim as optim


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def create_optimizer(args, parameters):
    param_groups = parameters

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=False
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    else:
        raise ValueError("Not a valid optimizer")

    return optimizer


def create_optimizer_fc(args, lr, parameters):
    param_groups = parameters

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=False
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    else:
        raise ValueError("Not a valid optimizer")

    return optimizer
