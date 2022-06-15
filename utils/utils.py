import os
import cv2
import random
from cv2 import cvtColor
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def for_directory(direc, cnt=0):
    if not os.path.exists(direc):
        os.makedirs(direc)
        os.chdir(direc)
    elif os.path.exists(direc):
        cnt += 1
        direc = direc + '_copy'
        for_directory(direc, cnt)


def save_parameter(input_size, model_name):
    f = open('./parameter' + '.txt', 'w')
    f.write('img_size : ' + str(input_size) + '\n')
    f.write('category : ' + str(model_name) + '\n')
    f.close()


def random_shuffle(x, seed):
    random.seed(seed)
    random.shuffle(x)
    return x


def grid_image(np_images, gts, preds, n=4):
    figure = plt.figure(figsize=(12, 18))
    plt.rc('font', size=10)
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n ** 0.5))
    for idx, (img, gt, pred) in enumerate(zip(np_images, gts, preds)):
        title = f"gt: {gt}, pred: {pred}"
        plt.subplot(n_grid, n_grid, idx+1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
    return figure


def custom_grid_image(np_images, main_gts, main_preds, sub_gts, sub_preds, n=4):
    figure = plt.figure(figsize=(12, 18))
    plt.rc('font', size=10)
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n ** 0.5))
    for idx, (img, main_gt, main_pred, sub_gt, sub_pred) in enumerate(zip(np_images, main_gts, main_preds, sub_gts, sub_preds)):
        title = f"main_gt: {main_gt}, main_pred: {main_pred}" + "\n" + f"sub_gt: {sub_gt}, sub_pred: {sub_pred}"
        plt.subplot(n_grid, n_grid, idx+1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
    return figure


def get_confusion_matrix(cfg, gt_lst, pred_lst):
    cf_matrix = confusion_matrix(gt_lst, pred_lst)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(cfg.DATASET.NUM_CLS)],
                         columns=[i for i in range(cfg.DATASET.NUM_CLS)])
    figure = plt.figure(figsize=(36, 22))
    plt.rc('font', size=30)
    sn.heatmap(df_cm, annot=True, fmt='d')
    return figure, df_cm


def get_confusion_matrix_torch(args, gt_lst, pred_lst):
    cf_matrix = confusion_matrix(gt_lst, pred_lst)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(args.class_n)],
                         columns=[i for i in range(args.class_n)])
    figure = plt.figure(figsize=(36, 22))
    plt.rc('font', size=30)
    sn.heatmap(df_cm, annot=True, fmt='d')
    return figure, df_cm


def calculate_metrics(args, df_cm):
    total = np.sum(df_cm.sum())
    df = pd.DataFrame({"Metrics": ['TP', 'FP', 'FN', 'TN', 'sensitivity',
                       'specificity', 'precision', 'f1 score', 'accuracy']})
    for cls in range(args.class_n):
        tmp = []
        tp = df_cm.loc[cls, cls]
        fp = np.sum(df_cm.loc[:, cls]) - tp
        fn = np.sum(df_cm.loc[cls]) - tp
        tn = total - tp - fp - fn

        tmp.append(tp)
        tmp.append(fp)
        tmp.append(fn)
        tmp.append(tn)

        sensitivity = round(tp/(tp+fn+0.00000000001), 5)
        specificity = round(tn/(tn+fp+0.00000000001), 5)
        precision = round(tp/(tp+fp+0.00000000001), 5)
        f1score = round(2*sensitivity*precision /
                        (sensitivity+precision+0.00000000001), 5)
        accuracy = round((tp+tn)/(tp+tn+fp+fn+0.00000000001), 5)

        tmp.append(round(sensitivity, 5))
        tmp.append(round(specificity, 5))
        tmp.append(round(precision, 5))
        tmp.append(round(f1score, 5))
        tmp.append(round(accuracy, 5))

        df[cls] = tmp

    df.columns = ["metrics", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0",
                  "5.5", "6.0", "6.5", "7.0", "7.5", "8.0", "8.5", "9.0", "9.5", "10.0",
                  "10.5", "11.0", "11.5", "12.0", "12.5", "13.0", "13.5", "14.0", "14.5",
                  "15.0", "15.5", "16.0", "16.5", "17.0", "17.5", "18.0"]
    return df


def make_mask(stop_lst):
    dic = {}
    for i, (s1, s2) in enumerate(zip(stop_lst[:-1], stop_lst[1:])):
        tmp = []
        for j in range(34):
            if s1 <= j < s2:
                tmp.append(False)
            else:
                tmp.append(True)
        dic[i] = tmp
    return dic
    