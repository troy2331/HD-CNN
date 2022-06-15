from os import pread
import random
import numpy as np

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchmetrics import F1Score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.dataset import class_dict
from train_utils.losses import L1Loss, SmoothL1Loss


f1_micro = F1Score(num_classes=34, average='micro')
f1_macro = F1Score(num_classes=34, average='macro')


transforms = [A.Compose([A.Resize(384, 384), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
              A.Compose([A.Resize(384, 384), A.CLAHE(p=1.), A.Normalize(
                  [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
              A.Compose([A.Resize(384, 384), A.InvertImg(p=1.), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()])]


def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # 연산 속도가 감소되는 문제가 있음
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def accuracy_fn(target, pred):
    target = target.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    accuracy = np.mean(np.cast[np.int32](np.equal(target, pred)))

    target_ = [class_dict[k] for k in target.numpy()]
    pred_ = [class_dict[k] for k in pred.numpy()]
    mae = mean_absolute_error(target_, pred_)
    f1_score_micro = f1_micro(pred, target)
    f1_score_macro = f1_macro(pred, target)
    return target, pred, accuracy, mae, f1_score_micro, f1_score_macro


def custom_accuracy_fn(main_target, main_pred, sub_target, sub_pred):
    main_target = main_target.cpu()
    sub_target = sub_target.cpu()
    main_pred = torch.argmax(main_pred, dim=1).cpu()
    sub_pred = torch.argmax(sub_pred, dim=1).cpu()

    main_accuracy = np.mean(np.cast[np.int32](
        np.equal(main_target, main_pred)))
    sub_accuracy = np.mean(np.cast[np.int32](
        np.equal(sub_target, sub_pred)))

    main_target_ = [class_dict[k] for k in main_target.numpy()]
    main_pred_ = [class_dict[k] for k in main_pred.numpy()]

    sub_target_ = [class_dict[k] for k in sub_target.numpy()]
    sub_pred_ = [class_dict[k] for k in sub_pred.numpy()]

    main_mae = mean_absolute_error(main_target_, main_pred_)
    sub_mae = mean_absolute_error(sub_target_, sub_pred_)

    main_f1_score_macro = f1_macro(main_pred, main_target)
    sub_f1_score_macro = f1_macro(sub_pred, sub_target)

    return main_target, main_pred, sub_target, sub_pred, main_accuracy, sub_accuracy, \
        main_mae, sub_mae, main_f1_score_macro, sub_f1_score_macro


def fc_accuracy_fn(pred_age, age, pred_cls, cls):
    pred_cls = torch.argmax(pred_cls, dim=1).cpu().detach().numpy()
    acc = np.mean(np.cast[np.int32](np.equal(pred_cls, cls.cpu().detach().numpy())))
    mse = mean_squared_error(
        pred_age.cpu().detach().numpy(), age.cpu().detach().numpy())
    return acc, mse


def train_step(batch_item, training, device, model, optimizer, criterion):
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)

    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img)
        if type(criterion) in [L1Loss, SmoothL1Loss]:
            loss = criterion(torch.argmax(
                output, dim=1).float(), label.float())
            loss.requires_grad_(True)
            loss.backward()
        else:
            loss = criterion(output, label)
            loss.backward()
        optimizer.step()
        target, pred, batch_acc, batch_mae, batch_f1_micro, batch_f1_macro = accuracy_fn(
            label, output)
        return loss, batch_acc, batch_mae, batch_f1_micro, batch_f1_macro, target, pred
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            if type(criterion) in [L1Loss, SmoothL1Loss]:
                loss = criterion(torch.argmax(
                    output, dim=1).float(), label.float())
            else:
                loss = criterion(output, label)

        target, pred, batch_acc, batch_mae, batch_f1_micro, batch_f1_macro = accuracy_fn(
            label, output)
        return loss, batch_acc, batch_mae, batch_f1_micro, batch_f1_macro, target, pred


def tta_step(batch_item, device, model, criterion):
    imgs = batch_item['raw_img']
    labels = batch_item['label'].unsqueeze(dim=1).to(device)
    tmps = torch.tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for img, label in zip(imgs, labels):
            tmp = torch.tensor(np.zeros((1, 34))).to(device)
            for transform in transforms:
                transformed_img = transform(image=img.cpu().numpy())[
                    'image'].to(device)
                output = model(transformed_img.unsqueeze(dim=0))
                tmp += output
            loss = criterion(tmp, label)
            tmps = torch.cat((tmps, tmp))

    target, pred, batch_acc, batch_mae, batch_f1_micro, batch_f1_macro = accuracy_fn(
        labels.squeeze(dim=1), tmps.squeeze(dim=1))
    return loss, batch_acc, batch_mae, batch_f1_micro, batch_f1_macro, target, pred


def custom_train_step(batch_item, training, device, fe_model, main_model, sub_model,
                      optimizer, criterion):
    img = batch_item['img'].to(device)
    main_label = batch_item['main_label'].to(device)
    sub_label = batch_item['sub_label'].to(device)
    mask = batch_item['mask'].to(device)

    if training is True:
        fe_model.train()
        main_model.train()
        sub_model.train()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            img_features = fe_model(img)
            main_pred = main_model(img_features)
            sub_pred = sub_model(img_features, mask)

        if type(criterion) in [L1Loss, SmoothL1Loss]:
            loss = criterion(torch.argmax(main_pred, dim=1).float(), main_label.float(
            )) + criterion(torch.argmax(sub_pred, dim=1).float(), sub_label.float())
            loss.requires_grad_(True)
            loss.backward()
        else:
            loss = (criterion(main_pred, main_label)) + \
                (criterion(sub_pred, sub_label))
            loss.backward()

        optimizer.step()
        main_target, main_pred, sub_target, sub_pred, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_f1_macro, batch_sub_f1_macro = custom_accuracy_fn(
            main_label, main_pred, sub_label, sub_pred)
        return loss, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_f1_macro, batch_sub_f1_macro, main_target, main_pred, sub_target, sub_pred
    else:
        fe_model.eval()
        main_model.eval()
        sub_model.eval()

        with torch.no_grad():
            img_features = fe_model(img)
            main_pred = main_model(img_features)
            sub_pred = sub_model(img_features, mask)

            if type(criterion) in [L1Loss, SmoothL1Loss]:
                loss = criterion(torch.argmax(main_pred, dim=1).float(), main_label.float(
                )) + criterion(torch.argmax(sub_pred, dim=1).float(), sub_label.float())
            else:
                loss = criterion(main_pred, main_label) + \
                    criterion(sub_pred, sub_label)

        main_target, main_pred, sub_target, sub_pred, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_f1_macro, batch_sub_f1_macro = custom_accuracy_fn(
            main_label, main_pred, sub_label, sub_pred)
        return loss, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_f1_macro, batch_sub_f1_macro, main_target, main_pred, sub_target, sub_pred


def custom_tta_step(batch_item, device, fe_model, main_model, sub_model, criterion, args):
    imgs = batch_item['raw_img']
    main_labels = batch_item['main_label'].unsqueeze(dim=1).to(device)
    sub_labels = batch_item['sub_label'].unsqueeze(dim=1).to(device)
    masks = batch_item['mask'].to(device)

    main_tmps = torch.tensor([]).to(device)
    sub_tmps = torch.tensor([]).to(device)

    fe_model.eval()
    main_model.eval()
    sub_model.eval()
    with torch.no_grad():
        for img, main_label, sub_label, mask in zip(imgs, main_labels, sub_labels, masks):
            main_tmp = torch.tensor(
                np.zeros((1, args.main_class_n))).to(device)
            sub_tmp = torch.tensor(np.zeros((1, args.class_n))).to(device)
            for transform in transforms:
                transformed_img = transform(image=img.cpu().numpy())[
                    'image'].to(device)
                img_feature = fe_model(transformed_img.unsqueeze(dim=0))
                main_pred = main_model(img_feature)
                sub_pred = sub_model(img_feature, mask)

                main_tmp += main_pred
                sub_tmp += sub_pred
            loss = criterion(main_tmp, main_label) + \
                criterion(sub_tmp, sub_label)
            main_tmps = torch.cat((main_tmps, main_tmp))
            sub_tmps = torch.cat((sub_tmps, sub_tmp))

    main_target, main_pred, sub_target, sub_pred, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_f1_macro, batch_sub_f1_macro = custom_accuracy_fn(
        main_labels.squeeze(dim=1), main_tmps.squeeze(dim=1), sub_labels.squeeze(dim=1), sub_tmps.squeeze(dim=1))
    return loss, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_f1_macro, batch_sub_f1_macro, main_target, main_pred, sub_target, sub_pred


def fc_train_step(alpha, batch_item, training, device, model, optimizer, reg_criterion, cls_criterion):
    prob = batch_item['prob'].to(device)
    age = batch_item['label'].to(device)
    cls = batch_item['cls'].to(device)

    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred_age, pred_cls = model(prob)

        loss = (reg_criterion(pred_age, age)*alpha) + \
            (cls_criterion(pred_cls, cls)*(1-alpha))
        loss.backward()

        optimizer.step()
        batch_acc, batch_mse = fc_accuracy_fn(pred_age, age, pred_cls, cls)
        return loss, batch_acc, batch_mse
    else:
        model.eval()
        with torch.no_grad():
            pred_age, pred_cls = model(prob)

        loss = (reg_criterion(pred_age, age)*alpha) + \
            (cls_criterion(pred_cls, cls)*(1-alpha))

        batch_acc, batch_mse = fc_accuracy_fn(pred_age, age, pred_cls, cls)
        return loss, batch_acc, batch_mse


def base_fc_train_step(args, batch_item, training, device, model, optimizer, criterion):
    prob = batch_item['prob'].to(device)
    if args.mode == 'regression':
        label = batch_item['label'].to(device)
    else:
        label = batch_item['cls'].to(device)

    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(prob)
        loss = criterion(pred, label)
        loss.backward()

        optimizer.step()
        if args.mode == 'regression':
            batch_metric = mean_squared_error(
                pred.cpu().detach().numpy(), label.cpu().detach().numpy())
        else:
            pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
            batch_metric = np.mean(np.cast[np.int32](
                np.equal(pred, label.cpu().detach().numpy())))
        return loss, batch_metric
    else:
        model.eval()
        with torch.no_grad():
            pred = model(prob)

        loss = criterion(pred, label)

        if args.mode == 'regression':
            batch_metric = mean_squared_error(
                pred.cpu().detach().numpy(), label.cpu().detach().numpy())
        else:
            pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
            batch_metric = np.mean(np.cast[np.int32](
                np.equal(pred, label.cpu().detach().numpy())))
        return loss, batch_metric
