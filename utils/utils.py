
#---------------------------------------------------#
#   本部分包括一些图像处理和权重选择的函数
#---------------------------------------------------#


import numpy as np
from PIL import Image


def get_lr(optimizer):      # 获得学习率
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):  # 图片归一化
    image /= 255.0  # 除255，归一化
    return image


def download_weights(backbone, model_dir="./model_data"):   # 当backbone不为vgg或者resnet时，应该不用
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


def resize_image(image, size):   #   对输入图像进行resize
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))   # 产生size大小的纯灰色图
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))  # 将输入图贴在上面产生的灰图上

    return new_image, nw, nh


def cvtColor(image):   # 将图像转换成RGB图像，防止灰度图在预测时报错。代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image