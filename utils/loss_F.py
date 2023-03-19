
#---------------------------------------------------#
#   本部分包含不同loss的计算方法
# 定义各类loss以及其他指标
# CE loss 图像分割中最常用的损失函数是逐像素交叉熵损失。该损失函数分别检查每个像素，将类预测(深度方向的像素向量)与热编码目标向量进行比较
# dice loss 通常用于计算两个样本的相似度 比较适用于样本极度不均的情况
# focal loss  整体上看，调节因子降低了 easy example 损失的贡献，调节了简单样本权重降低的速率，并拓宽了样本接收低损失的范围。对于困难样本的学习更有利。
#---------------------------------------------------#


import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)  # 这个函数是用来上采样或下采样，可以给定size或者scale_factor来进行上下采样

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)   # view（-1）动态调整该维度上的元素个数。  如果想要断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝。
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

