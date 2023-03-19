import os
import torch
import torch.nn as nn
import cv2
from PIL import Image
# from network.ACSNet import ACSNet
from nets.Mynet2_15 import mynet
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, module, extract_layer):
        super(FeatureExtractor, self).__init__()
        self.module = module
        self.extract_layer = extract_layer

    def forward(self, x):
        outputs = {}

        x1, x2, x3, x4 = self.module(x)
        outputs['out1'] = x1
        outputs['out2'] = x2
        outputs['out3'] = x3
        outputs['out4'] = x4
        # outputs['out5'] = x5
        # outputs['out6'] = x6
        #outputs['out7'] = x7
        #outputs['out8'] = x8
        #outputs['out9'] = x9
        #outputs['out10'] = x10

        return outputs


def get_feature():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pic_dir = 'E:/anaconda\envs\pythonProject\CODE\MyCode\Data\Dataset\JPEGImages/06_045.jpg'  # 插入图片
    img1 = Image.open(pic_dir).convert('RGB')
    img2 = img1.resize((512, 512), Image.ANTIALIAS)
    img3 = np.array(img2, dtype=np.uint8)
    # [N, C, H, W]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = data_transform(img3)
    img4 = torch.unsqueeze(img_tensor, dim=0)

    net = mynet(num_classes=6).to(device)
    net.load_state_dict(torch.load('E:/anaconda\envs\pythonProject\CODE\MyCode\model_data\mynet/ep200-loss0.064-val_loss0.087.pth'))
    net.eval()
    exact_list = None
    dst = 'E:/anaconda\envs\pythonProject\CODE\MyCode\Data\Dataset/picinprocess'  # 保存的路径
    size1, size2 = 512, 512  # 放大尺寸

    myexactor = FeatureExtractor(net, exact_list)
    output = myexactor(img4.to(device))
    for k, v in output.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            dst_path = os.path.join(dst, k)
            if os.path.exists(dst_path) is False:
                os.makedirs(dst_path)

            heatmap = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            tmp_file = os.path.join(dst_path, str(i) + '.png')
            heatmap = cv2.resize(heatmap, (size1, size2), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(tmp_file, heatmap)
        # heatmap = sum(heatmap, 2)
        # print(heatmap.shape)
        # plt.imshow(heatmap)
        # plt.show()


if __name__ == '__main__':
    get_feature()


