import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = torch.log(torch.softmax(input, dim=1))  #
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DatasetClass(Dataset):
    """继承Dataset类，重写__getitem__和__len__方法
    添加get_idx方法，返回id
    input: wavlabel, wavdata, wavidx
    """

    # Initialize your data, download, etc.
    def __init__(self, features, wav_label, wav_index):
        # 直接传递data和label
        # self.len = wavlen
        # embeds = []
        # for embed in wavebd:
        #     embed = int(embed.split('.')[0])
        #     embeds.append(embed)
        # self.wavebd = embeds
        self.data = torch.from_numpy(features)
        self.label = torch.from_numpy(wav_label)
        self.idx = torch.from_numpy(wav_index)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        label_item = self.label[index]
        idx_item = self.idx[index]
        # embeding = self.wavebd[index]
        # embeding = 1  # fake
        # wide_feat = hand_fea((data_item, 4000))
        return data_item.float(), label_item, idx_item  # , wide_feat, embeding

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)

    # def get_idx(self, index):
    #     iditem = self.id[index]
    #     return iditem


class BCEFocalLoss(nn.Module):
    """BiFocal Loss"""

    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        self.gamma = self.gamma.view(target.size)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
                1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLoss_VGG(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True, weight=None):
        super(FocalLoss_VGG, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight
        # if isinstance(alpha, list):
        #     assert len(alpha) == num_classes
        #     self.alpha = torch.Tensor(alpha)
        # else:
        #     assert alpha < 1
        #     self.alpha = torch.zeros(num_classes)
        #     self.alpha[0] += alpha
        #     self.alpha[1:] += (1 - alpha)

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                torch.argmax(inputs, dim=1).float(), targets, weight=self.weight, reduce=False)
        else:
            CE_loss = nn.CrossEntropyLoss(
                inputs, targets, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class MyDataset(Dataset):
    """my dataset."""

    # Initialize your data, download, etc.
    def __init__(self, wavlabel, wavdata):
        # 直接传递data和label
        # self.len = wavlen
        self.data = torch.tensor(wavdata)
        self.label = torch.tensor(wavlabel)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        dataitem = self.data[index]
        labelitem = self.label[index]
        return dataitem.float(), labelitem.float()

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)


class DatasetClass_t(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, wavlabel, wavdata, wavidx):
        # 直接传递data和label
        # self.len = wavlen
        # embeds = []
        # for embed in wavebd:
        #     embed = int(embed.split('.')[0])
        #     embeds.append(embed)
        # self.wavebd = embeds
        self.data = torch.from_numpy(wavdata)
        self.label = torch.from_numpy(wavlabel)
        self.id = torch.from_numpy(wavidx)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        dataitem = self.data[index]
        labelitem = self.label[index]
        iditem = self.id[index]
        # embeding = self.wavebd[index]
        embeding = 1  # fake
        # wide_feat = hand_fea((dataitem, 4000))
        return dataitem.float(), labelitem, iditem

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)
