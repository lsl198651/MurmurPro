import argparse
import logging

import numpy as np
import torch.profiler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from model.senet.my_resnet import My_ResNet
from train_eval import train_test
from util.class_def import DatasetClass
from util.utils_dataloader import fold5_dataloader
from util.utils_train import logger_init

# from util.dataloaders import get_features
# from model.model_sknet import AudioClassifier
# from BEATs import BEATs_Pre_Train_itere3
# from torchvision.models import resnet18

#
if __name__ == '__main__':
    # ========================/ 函数入口 /========================== #
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=512, help="args.batch_size for training")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning_rate for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
    parser.add_argument("--layers", type=int, default=3, help="layers number")
    parser.add_argument("--loss_type", type=str, default="FocalLoss", help="loss function", choices=["CE", "FocalLoss"])
    parser.add_argument("--scheduler_flag", type=str, default="step", help="the dataset used",
                        choices=["cos", "MultiStepLR", "step"], )
    parser.add_argument("--freqm_value", type=int, default=0, help="frequency mask max length")
    parser.add_argument("--timem_value", type=int, default=0, help="time mask max length")
    parser.add_argument("--mask", type=bool, default=False, help="number of classes", choices=[True, False])
    parser.add_argument("--trainset_balence", type=bool, default=False, help="balance absent and present in testset",
                        choices=[True, False], )
    parser.add_argument("--Data_Augmentation", type=bool, default=False, help="Add data augmentation",
                        choices=[True, False], )
    parser.add_argument("--train_total", type=bool, default=True, help="use grad_no_requiredn", choices=[True, False], )
    parser.add_argument("--samplerWeight", type=bool, default=True, help="use balanced sampler",
                        choices=[True, False], )
    # TODO 改模型名字
    parser.add_argument("--model", type=str,
                        default="logmel + se_resnet6v2  4k  samplerWeight[1,5] lr=0.05,32,64 channel reductiom=8 high_freq=1000, window_type='hanning'")
    parser.add_argument("--ap_ratio", type=float, default=1.0, help="ratio of absent and present")
    parser.add_argument("--beta", type=float, default=(0.9, 0.98), help="beta")
    parser.add_argument("--cross_evalue", type=bool, default=False)
    parser.add_argument("--train_fold", type=list, default=['0', '1', '2', '3'])
    parser.add_argument("--test_fold", type=list, default=['4'])
    parser.add_argument("--setType", type=str, default=r"\13_baseset_4s_4k")
    parser.add_argument("--model_folder", type=str,
                        default=r"D:\Shilong\murmur\00_Code\LM\beats1\SE_ResNet6\MyModels")
    parser.add_argument("--isTensorboard", type=bool, default=False)
    parser.add_argument("--isTry", type=bool, default=True)
    args = parser.parse_args()
    all_list = ['0', '1', '2', '3', '4']

    # ========================/ 选择模型 /========================== #
    MyModel = My_ResNet(layers=[1, 1])
    # ========================/ 设置优化器 /========================== #
    if not args.train_total:
        for param in MyModel.BEATs.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MyModel.parameters()),
                                      lr=args.learning_rate,
                                      betas=args.beta, )
    else:
        optimizer = torch.optim.AdamW(MyModel.parameters(),
                                      lr=args.learning_rate,
                                      betas=args.beta)

    # ========================/ 检测分折重复 /========================== #
    for val in args.test_fold:
        if val in args.train_fold:
            raise ValueError("train_fold and test_fold have same fold")

    if args.isTry is True:
        train_fold = '0'
        test_fold = '1'
    else:
        for fold in all_list:
            train_fold = all_list.remove(fold)
            test_fold = fold
    # ========================/ 加载数据集 /========================== #
    train_features, train_label, train_index, test_features, test_label, test_index = fold5_dataloader(
        args.train_fold, args.test_fold, args.Data_Augmentation, args.setType)
    # ========================/ setup loader /========================== #
    if args.samplerWeight == True:
        weights = [5 if label == 1 else 1 for label in train_label]
        Data_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(DatasetClass(wavlabel=train_label, wavdata=train_features, wavidx=train_index),
                                  sampler=Data_sampler,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=2)
    else:
        train_loader = DataLoader(DatasetClass(wavlabel=train_label, wavdata=train_features, wavidx=train_index),
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=2)

    val_loader = DataLoader(DatasetClass(wavlabel=test_label, wavdata=test_features, wavidx=test_index),
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2)

    # ========================/ 计算数据集大小 /========================== #
    train_present_size = np.sum(train_label == 1)
    train_absent_size = np.sum(train_label == 0)
    test_present_size = np.sum(test_label == 1)
    test_absent_size = np.sum(test_label == 0)
    train_set_size = train_label.shape[0]
    test_set_size = test_label.shape[0]

    # ========================/ 打印日志 /========================== #
    # import torch
    # print(torch.__version__)
    logger_init()
    logging.info(f"{args.model}  ")
    logging.info(f"# Batch_size = {args.batch_size}")
    logging.info(f"# Num_epochs = {args.num_epochs}")
    logging.info(f"# Learning_rate = {args.learning_rate:.1e}")
    logging.info(f"# lr_scheduler = {args.scheduler_flag}")
    logging.info(f"# Loss_fn = {args.loss_type}")
    # logging.info(f"# Data Augmentation = {args.Data_Augmentation}")
    # logging.info(f"# Trainset_balance = {args.trainset_balence}")
    # logging.info(f"# train_total = {args.train_total}")
    # logging.info(f"# Masking = {args.mask}")
    logging.info(f"# SetType = {args.setType}")
    logging.info(f"# Train_a/p = {train_absent_size}/{train_present_size}")
    logging.info(f"# Test_a/p = {test_absent_size}/{test_present_size}")
    logging.info(f"# Trainset_size = {train_set_size}")
    logging.info(f"# Testset_size = {test_set_size}")
    logging.info(f"# Train_fold = {args.train_fold}")
    logging.info(f"# Test_fold = {args.test_fold}")
    logging.info("# Optimizer = " + str(optimizer))
    logging.info("# ")
    # ========================/ 开始训练 /========================== #

    train_test(model=MyModel,
               train_loader=train_loader,
               val_loader=val_loader,
               optimizer=optimizer,
               args=args)
