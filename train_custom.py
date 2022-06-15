from utils.utils import for_directory, get_confusion_matrix_torch, custom_grid_image
from train_utils.schedulers import create_scheduler
from train_utils.optimizers import create_optimizer, get_lr
from train_utils.losses import create_criterion
from models.custom_model import FeatureExtractor, MainClassifier, SubClassifier
from datasets.BAdataset import BADataset_custom, train_transforms, test_transforms
from trainer import seed_everything, custom_train_step, custom_tta_step
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import wandb
import matplotlib
matplotlib.use('Agg')


def main(args):
    seed_everything(args.random_seed)

    # device
    if args.device == 'gpu0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    elif args.device == 'gpu1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # save directory
    i = datetime.now()
    current_date = i.strftime('%Y/%m/%d').replace('/', '_')
    directory_name = os.path.join(
        args.save_path, current_date, args.data_path.split('/')[-1])
    for_directory(directory_name)

    # dataset & dataloader
    train_transform = train_transforms(args)
    test_transform = test_transforms(args)

    train_dataset = BADataset_custom(
        args, transforms=train_transform, mode='train', target_label='total')
    val_dataset = BADataset_custom(
        args, transforms=test_transform, mode='valid', target_label='total')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # model
    fe_model = FeatureExtractor()
    fe_model = fe_model.to(device)
    main_model = MainClassifier(main_classes=args.main_class_n)
    main_model = main_model.to(device)
    sub_model = SubClassifier()
    sub_model = sub_model.to(device)

    # loss
    criterion = create_criterion(args)

    # optimizer & scheduler
    # optimizer = create_optimizer(args, model.parameters())
    # optimizer = torch.optim.Adam([
    #     {'params': fe_model.parameters()},
    #     {'params': main_model.parameters()},
    #     {'params': sub_model.parameters()},
    # ], lr=args.lr, weight_decay=args.weight_decay)
    optimizer = create_optimizer(args, [
        {'params': fe_model.parameters()},
        {'params': main_model.parameters()},
        {'params': sub_model.parameters()},
    ])
    scheduler = create_scheduler(args, optimizer)

    val_loss_plot, val_main_metric_plot, val_sub_metric_plot = [], [], []
    step = 0
    for epoch in range(args.epochs):
        total_loss, total_val_loss = 0, 0
        total_main_acc, total_val_main_acc = 0, 0
        total_main_mae, total_val_main_mae = 0, 0
        total_main_macro, total_val_main_macro = 0, 0
        total_sub_acc, total_val_sub_acc = 0, 0
        total_sub_mae, total_val_sub_mae = 0, 0
        total_sub_macro, total_val_sub_macro = 0, 0

        training = True
        print("training...")
        tqdm_dataset = tqdm(enumerate(train_loader))
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_macro, batch_sub_macro, _, _, _, _ = custom_train_step(
                batch_item, training, device, fe_model, main_model, sub_model, optimizer, criterion)
            total_loss += batch_loss
            total_main_acc += batch_main_acc
            total_main_mae += batch_main_mae
            total_main_macro += batch_main_macro
            total_sub_acc += batch_sub_acc
            total_sub_mae += batch_sub_mae
            total_sub_macro += batch_sub_macro

            current_lr = get_lr(optimizer)
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'lr': '{:06f}'.format(current_lr),
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Loss': '{:06f}'.format(total_loss/(batch+1)),
                # 'Mean Acc': '{:06f}'.format(total_acc/(batch+1)),
                # 'Mean MAE': '{:06f}'.format(total_mae/(batch+1))
            })
        if args.wandb == True:
            wandb.log(
                {
                    "Train/Train loss": total_loss/(batch+1),
                    "Train/Train main acc": total_main_acc/(batch+1),
                    "Train/Train main MAE": total_main_mae/(batch+1),
                    "Train/Train main f1 macro": total_main_macro/(batch+1),
                    "Train/Train sub acc": total_sub_acc/(batch+1),
                    "Train/Train sub MAE": total_sub_mae/(batch+1),
                    "Train/Train sub f1 macro": total_sub_macro/(batch+1),
                    "Train/Learning rate": current_lr
                },
                step=step
            )

        training = False
        print("Validation...")
        total_main_target, total_main_pred = [], []
        total_sub_target, total_sub_pred = [], []
        tqdm_dataset = tqdm(enumerate(val_loader))
        for batch, batch_item in tqdm_dataset:
            figure = None

            if args.tta == True:
                batch_loss, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_macro, batch_sub_macro, batch_main_target, batch_main_pred, batch_sub_target, batch_sub_pred = custom_tta_step(
                    batch_item, device, fe_model, main_model, sub_model, criterion, args)
                total_main_target.extend(batch_main_target)
                total_main_pred.extend(batch_main_pred)
                total_sub_target.extend(batch_sub_target)
                total_sub_pred.extend(batch_sub_pred)
                total_val_loss += batch_loss
                total_val_main_acc += batch_main_acc
                total_val_main_mae += batch_main_mae
                total_val_main_macro += batch_main_macro
                total_val_sub_acc += batch_sub_acc
                total_val_sub_mae += batch_sub_mae
                total_val_sub_macro += batch_sub_macro

                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Val Loss': '{:06f}'.format(total_val_loss/(batch+1)),
                    # 'Mean Val Acc': '{:06f}'.format(total_val_acc/(batch+1)),
                    # 'Mean Val MAE': '{:06f}'.format(total_val_mae/(batch+1))
                })
                figure = custom_grid_image(
                    batch_item['raw_img'], batch_main_target, batch_main_pred, batch_sub_target, batch_sub_pred, n=args.batch_size)
                if args.wandb == True:
                    wandb.log(
                        {
                            "Media/Predict Images": figure
                        },
                        step=step
                    )
                    step += 1
                plt.close(figure)
            else:
                batch_loss, batch_main_acc, batch_sub_acc, batch_main_mae, batch_sub_mae, batch_main_macro, batch_sub_macro, batch_main_target, batch_main_pred, batch_sub_target, batch_sub_pred = custom_train_step(
                    batch_item, training, device, fe_model, main_model, sub_model, optimizer, criterion)
                total_main_target.extend(batch_main_target)
                total_main_pred.extend(batch_main_pred)
                total_sub_target.extend(batch_sub_target)
                total_sub_pred.extend(batch_sub_pred)
                total_val_loss += batch_loss
                total_val_main_acc += batch_main_acc
                total_val_main_mae += batch_main_mae
                total_val_main_macro += batch_main_macro
                total_val_sub_acc += batch_sub_acc
                total_val_sub_mae += batch_sub_mae
                total_val_sub_macro += batch_sub_macro

                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Val Loss': '{:06f}'.format(total_val_loss/(batch+1)),
                    # 'Mean Val Acc': '{:06f}'.format(total_val_acc/(batch+1)),
                    # 'Mean Val MAE': '{:06f}'.format(total_val_mae/(batch+1))
                })
                figure = custom_grid_image(
                    batch_item['raw_img'], batch_main_target, batch_main_pred, batch_sub_target, batch_sub_pred, n=args.batch_size)
                if args.wandb == True:
                    wandb.log(
                        {
                            "Media/Predict Images": figure
                        },
                        step=step
                    )
                    step += 1
                plt.close(figure)

        val_loss_plot.append(total_val_loss/(batch+1))
        val_main_metric_plot.append(round(total_val_main_mae/(batch+1), 4))
        val_sub_metric_plot.append(round(total_val_sub_mae/(batch+1), 4))

        args.class_n = 34
        cm1, df_cm = get_confusion_matrix_torch(
            args, total_sub_target, total_sub_pred)
        args.class_n = args.main_class_n
        cm2, df_cm = get_confusion_matrix_torch(
            args, total_main_target, total_main_pred)
        args.class_n = 34
        # df = calculate_metrics(args, df_cm)

        if args.wandb == True:
            wandb.log(
                {
                    "Media/Main Confusion Matrix": wandb.Image(cm2),
                    "Media/Sub Confusion Matrix": wandb.Image(cm1),
                    # "Metric/Metrics": df,
                    "Valid/Valid loss": total_val_loss/(batch+1),
                    "Valid/Valid main acc": total_val_main_acc/(batch+1),
                    "Valid/Valid main MAE": total_val_main_mae/(batch+1),
                    "Valid/Valid main f1 macro": total_val_main_macro/(batch+1),
                    "Valid/Valid sub acc": total_val_sub_acc/(batch+1),
                    "Valid/Valid sub MAE": total_val_sub_mae/(batch+1),
                    "Valid/Valid sub f1 macro": total_val_sub_macro/(batch+1)
                },
                step=step
            )
        plt.close(cm1)
        plt.close(cm2)

        scheduler.step()

        if np.min(val_sub_metric_plot) == val_sub_metric_plot[-1]:
            # save path
            fe_save_path = directory_name + \
                f'/fe_{args.data_path.split("/")[-1]}_{epoch}_{val_sub_metric_plot[-1]}.pt'
            main_save_path = directory_name + \
                f'/main_{args.data_path.split("/")[-1]}_{epoch}_{val_sub_metric_plot[-1]}.pt'
            sub_save_path = directory_name + \
                f'/sub_{args.data_path.split("/")[-1]}_{epoch}_{val_sub_metric_plot[-1]}.pt'
            torch.save(fe_model.state_dict(), fe_save_path)
            torch.save(main_model.state_dict(), main_save_path)
            torch.save(sub_model.state_dict(), sub_save_path)
            print("Saved model weight!! {}".format(fe_save_path))

    if args.wandb == True:
        wandb.alert(
            title="Finish!",
            text="The training is over",
            level=wandb.AlertLevel.INFO
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_seed", type=int, default=2022, help="random seed (default: 2022)")
    parser.add_argument(
        "--device", type=str, default="gpu1", help="device type (default: gpu1)")

    # dataset
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="input batch size for training (default: 16)")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers for training (default: 4)")
    parser.add_argument(
        "--img_size", type=int, default=224, help="image size (default: 224)")
    parser.add_argument(
        "--padding", action="store_true", default="False", help="padding implement or not")

    # model
    parser.add_argument(
        "--main_class_n", type=int, default=6, help="main class size for training (default: 6)")
    parser.add_argument(
        "--class_n", type=int, default=34, help="class size for training (default: 34)")
    parser.add_argument(
        "--model", type=str, help="model type")

    # criterion
    parser.add_argument(
        "--criterion", type=str, default="cross_entropy", help="criterion type (default: cross_entropy)")
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="label smoothing value (default: 0.1)")
    parser.add_argument(
        "--adjacency", action="store_true", help="adjacency implement or not in label smoothing")

    # optimizer
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer type (default: adam)")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay (default: 1e-5)")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum (default: 0.9)")
    parser.add_argument(
        "--amsgrad", action="store_true", help="amsgrad for adam")

    # scheduler
    parser.add_argument(
        "--scheduler", type=str, default="lambda", help="scheduler type (default: lambda)",)
    parser.add_argument(
        "--poly_exp", type=float, default=1.0, help="polynomial LR exponent (default: 1.0)",)
    parser.add_argument(
        "--T_max", type=int, default=10, help="cosineannealing T_max (default: 10)")
    parser.add_argument(
        "--eta_min", type=int, default=0, help="cosineannealing eta_min (default: 0)")
    parser.add_argument(
        "--step_size", type=int, default=10, help="stepLR step_size (default: 10)")
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="stepLR gamma (default: 0.1)")
    parser.add_argument(
        "--max_lr", type=float, default=1e-3, help="OneCycleLR max lr (default: 1e-3)")
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="min lr (default: 1e-5)")
    parser.add_argument(
        "--t_up", type=int, default=3, help="warm up epoch (default: 3)")
    parser.add_argument(
        "--iter", type=int, default=10, help="iteration rate (default: 10)")

    # TTA(Test Time Augmentation)
    parser.add_argument(
        "--tta", action="store_true", default="True", help="tta implement or not")

    # Container environment
    parser.add_argument(
        "--data_path", type=str, default=os.path.abspath('./data/radius'), help="dataset path")
    parser.add_argument(
        "--save_path", type=str, default=os.path.abspath('./save_model/custom/'), help="model save dir path")
    parser.add_argument(
        "--train_csv", type=str, default=os.path.abspath('./data/radius_train.csv'), help="train csv path")
    parser.add_argument(
        "--valid_csv", type=str, default=os.path.abspath('./data/radius_valid.csv'), help="valid csv path")

    # wandb
    parser.add_argument(
        "--wandb", action="store_true", default="True", help="wandb implement or not")
    parser.add_argument(
        "--entity", type=str, default="troy2331", help="wandb entity name (default: jaehwan)",)
    parser.add_argument(
        "--project", type=str, default="BoneAge_custom_mp_tta", help="wandb project name (default: torch)")

    args = parser.parse_args()
    print(args)

    # wandb init
    if args.wandb == True:
        wandb.init(entity=args.entity, project=args.project)
        wandb.run.name = 'custom_model'

    main(args)
