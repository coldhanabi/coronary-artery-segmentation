
#---------------------------------------------------#
#   本部分分割数据集为训练集、验证集和测试集
#---------------------------------------------------#



import os
import random

# ----------------------------------------------------------------------#
#   train:val:test = 8:1:1
# ----------------------------------------------------------------------#

def divide_data (Data_path, trainval_percent, train_percent):
    imgfilepath = os.path.join(Data_path, 'Dataset/JPEGImages')
    saveBasePath = os.path.join(Data_path, 'Dataset/ImageSets/Segmentation')
    temp_img = os.listdir(imgfilepath)
    total_img = []
    for img in temp_img:
        if img.endswith('.jpg'):
            total_img.append(img)

    num = len(total_img)  # num是数据集样本标签总数
    list = range(num)
    tv = int(num * trainval_percent)  # train+val 总数
    tr = int(num * train_percent)  # train 总数
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_img[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    return

divide_data('Data', 0.9, 0.8)


