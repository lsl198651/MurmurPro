import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_auprc, binary_auroc, binary_f1_score, binary_confusion_matrix, \
    binary_accuracy, binary_precision, binary_recall
from transformers import optimization

from util.class_def import FocalLoss
from util.utils_train import new_segment_classifier


def train_test(model,
               train_loader,
               val_loader,
               optimizer=None,
               args=None):
    global lr_now, scheduler
    # ========================/ 声明 /========================== #
    error_index_path = r"./error_index/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    patient_error_index_path = r"./patient_error_index/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    if not os.path.exists(error_index_path):
        os.makedirs(error_index_path)
    if not os.path.exists(patient_error_index_path):
        os.makedirs(patient_error_index_path)
    tb_writer = SummaryWriter(r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
    confusion_matrix_path = r"./confusion_matrix/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    lr = []
    max_test_acc = []
    max_train_acc = []
    best_ACC = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    model = model.to(device)  # 放到设备中
    # for amp
    # ========================/ 学习率设置 /========================== #
    # scaler = GradScaler()
    warm_up_ratio = 0.1
    total_steps = len(train_loader) * args.num_epochs
    if args.scheduler_flag == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif args.scheduler_flag == "cos_warmup":
        scheduler = optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=warm_up_ratio * total_steps,
                                                                 num_training_steps=total_steps)
    elif args.scheduler_flag == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
    elif args.scheduler_flag == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 80], gamma=0.1)
    # ========================/ 损失函数 /========================== #

    if args.loss_type == "FocalLoss":
        loss_fn = FocalLoss()
    elif args.loss_type == "CE_weighted":
        normed_weights = [1, 5]
        normed_weights = torch.FloatTensor(normed_weights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=normed_weights)  # 内部会自动加上Softmax层,weight=normedWeights
    else:
        loss_fn = nn.CrossEntropyLoss()
    # ========================/ 训练网络 /========================== #
    for epochs in range(args.num_epochs):
        # train model
        model.train()
        train_loss = 0
        correct_t = 0
        train_len = 0
        input_train = []
        target_train = []
        for data_t, label_t, index_t in train_loader:
            data_t, label_t, index_t = data_t.to(device), label_t.to(device), index_t.to(device)
            predict_t = model(data_t)
            loss = loss_fn(predict_t, label_t.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # get the index of the max log-probability
            pred_t = predict_t.max(1, keepdim=True)[1]
            pred_t = pred_t.squeeze(1)
            input_train.extend(pred_t.cpu().tolist())
            target_train.extend(label_t.cpu().tolist())
            correct_t += pred_t.eq(label_t).sum().item()
            train_len += len(pred_t)
        # ------------------调库计算指标--------------------------
        train_input, train_target = torch.as_tensor(
            input_train), torch.as_tensor(target_train)
        train_acc = binary_accuracy(train_input, train_target)
        # print(f"train_acc:{train_acc:.2%}")
        # ========================/ 验证网络 /========================== #
        model.eval()
        label = []
        pred = []
        error_index = []
        result_list_present = []
        test_loss = 0
        correct_v = 0
        with torch.no_grad():
            for data_v, label_v, index_v in val_loader:
                data_v, label_v, index_v = data_v.to(device), label_v.to(device), index_v.to(device)
                optimizer.zero_grad()
                predict_v = model(data_v)
                loss_v = loss_fn(predict_v, label_v.long())
                # get the index of the max log-probability
                pred_v = predict_v.max(1, keepdim=True)[1]
                test_loss += loss_v.item()
                pred_v = pred_v.squeeze(1)
                correct_v += pred_v.eq(label_v).sum().item()
                idx_v = index_v[pred_v.ne(label_v)]
                result_list_present.extend(index_v[pred_v.eq(1)].cpu().tolist())
                try:
                    error_index.extend(idx_v.cpu().tolist())
                except TypeError:
                    print("TypeError: 'int' object is not iterable")
                pred.extend(pred_v.cpu().tolist())
                label.extend(label_v.cpu().tolist())
        if args.scheduler_flag is not None:
            scheduler.step()
        # ------------------调库计算指标--------------------------
        test_input, test_target = torch.as_tensor(pred), torch.as_tensor(label)
        segments_AUPRC = binary_auprc(test_input, test_target)
        segments_AUROC = binary_auroc(test_input, test_target)
        test_acc = binary_accuracy(test_input, test_target)
        segments_F1 = binary_f1_score(test_input, test_target)
        segments_CM = binary_confusion_matrix(test_input, test_target)
        segments_PPV = binary_precision(test_input, test_target)
        segments_TPR = binary_recall(test_input, test_target)
        # --------------------------------------------------------
        # pd.DataFrame(error_index).to_csv(error_index_path + "/epoch" + str(epochs + 1) + ".csv",
        #                                  index=False,
        #                                  header=False)
        result, target = new_segment_classifier(result_list_present, args.test_fold)
        test_patient_input, test_patient_target = torch.as_tensor(result), torch.as_tensor(target)
        patient_AUPRC = binary_auprc(test_patient_input, test_patient_target)
        patient_AUROC = binary_auroc(test_patient_input, test_patient_target)
        patient_ACC = binary_accuracy(test_patient_input, test_patient_target)
        patient_F1 = binary_f1_score(test_patient_input, test_patient_target)
        patient_CM = binary_confusion_matrix(test_patient_input, test_patient_target)
        # # 这两个算出来的都是present的
        patient_PPV = binary_precision(test_patient_input, test_patient_target)
        patient_TPR = binary_recall(test_patient_input, test_patient_target)
        "保存最好的模型"
        if patient_ACC > best_ACC:
            best_ACC = patient_ACC
        # if  args.saveModel is True:
        #     save_checkpoint({"epoch": epochs + 1,
        #                      "model": model.state_dict(),
        #                      "optimizer": optimizer.state_dict()},
        #                     "se_resnet6v2",
        #                     args.test_fold[0],
        #                     "{}".format(args.model_folder))
        # ========================/ 保存error_index.csv  /========================== #
        # pd.DataFrame(patient_error_id).to_csv(patient_error_index_path + "/epoch" + str(epochs + 1) + ".csv",
        #                                       index=False,
        #                                       header=False)
        for group in optimizer.param_groups:
            lr_now = group["lr"]
        lr.append(lr_now)
        # "更新权值"
        test_loss /= len(pred)
        train_loss /= train_len
        max_train_acc.append(train_acc)
        max_test_acc.append(test_acc)
        max_train_acc_value = max(max_train_acc)
        max_test_acc_value = max_test_acc[max_train_acc.index(max_train_acc_value)]
        # ********************** tensorboard绘制曲线 ********************** #
        if args.isTensorboard:
            tb_writer.add_scalar("train_acc", train_acc, epochs)
            tb_writer.add_scalar("test_acc", test_acc, epochs)
            tb_writer.add_scalar("train_loss", train_loss, epochs)
            tb_writer.add_scalar("test_loss", test_loss, epochs)
            tb_writer.add_scalar("learning_rate", lr_now, epochs)
            # tb_writer.add_scalar("patient_acc", test_patient_acc, epochs)
        # ========================/ 日志 /========================== #
        logging.info(f"============================")
        logging.info(f"EPOCH: {epochs + 1}/{args.num_epochs}")
        logging.info(f"Learning rate: {lr_now:.1e}")
        logging.info(f"LOSS: train:{train_loss:.2e} verify:{test_loss:.2e}")
        logging.info(f"segments_CM:{segments_CM.numpy()}")
        logging.info(f"ACC: train:{train_acc:.2%} verify:{test_acc:.2%}")
        logging.info(f"segments_TPR:{segments_TPR:.2%}")
        logging.info(f"segments_PPV:{segments_PPV:.2%}")
        logging.info(f"segments_F1:{segments_F1:.2%}")
        logging.info(f"segments_AUROC:{segments_AUROC:.2%}")
        logging.info(f"segments_AUPRC:{segments_AUPRC:.2%}")
        logging.info(f"----------------------------")
        logging.info(f"patient_CM:{patient_CM.numpy()}")
        logging.info(f"patient_ACC:{patient_ACC:.2%}")
        logging.info(f"patient_TPR:{patient_TPR:.2%}")
        logging.info(f"patient_PPV:{patient_PPV:.2%}")
        logging.info(f"patient_F1:{patient_F1:.2%}")
        logging.info(f"patient_AUROC:{patient_AUROC:.2%}")
        logging.info(f"patient_AUPRC:{patient_AUPRC:.2%}")
        logging.info(f"LR max:{max(lr):.1e} min:{min(lr):.1e}")
        logging.info(f"ACC MAX train:{max_train_acc_value:.2%} verify:{max_test_acc_value:.2%}")
        logging.info(f"best ACC:{best_ACC:.2%}")

        # ========================/ 混淆矩阵 /========================== #
        """draw_confusion_matrix(
            test_cm.numpy(),
            ["Absent", "Present"],
            "epoch" + str(epochs + 1) + ",testacc: {:.3%}".format(test_acc),
            pdf_save_path=confusion_matrix_path,
            epoch=epochs + 1
        )"""
