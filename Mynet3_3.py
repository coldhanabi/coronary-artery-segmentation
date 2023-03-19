


import torch
import torch.nn as nn
import nets.Block


class mynet(nn.Module):
    def __init__(self, num_classes=6, pretrained=False, backbone='vgg'):
        super(mynet, self).__init__()
        self.pretrained = pretrained

        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        self.layer1 = nets.Block.BasicBlock(32, 32)
        self.layer2 = nets.Block.BasicBlock(64, 64)
        self.layer3 = nets.Block.BasicBlock(128, 128)
        self.layer4 = nets.Block.BasicBlock(256, 256)
        self.layer5 = nets.Block.BasicBlock(512, 512)

        self.layer_1 = nets.Block.BasicBlock(32, 32)
        self.layer_2 = nets.Block.BasicBlock(32, 64)
        self.layer_3 = nets.Block.BasicBlock(64, 128)
        self.layer_4 = nets.Block.BasicBlock(128, 256)

        self.upblock1 = nets.Block.upblock(128, 32)
        self.upblock2 = nets.Block.upblock(256, 64)
        self.upblock3 = nets.Block.upblock(512, 128)
        self.upblock4 = nets.Block.upblock(1024, 256)

        self.reblock1 = nets.Block.RE(128, 64)
        self.reblock2 = nets.Block.RE(256, 128)
        self.reblock3 = nets.Block.RE(512, 256)
        self.reblock4 = nets.Block.RE(512, 512)

        self.mcse1 = nets.Block.MCSE26(64)
        self.mcse2 = nets.Block.MCSE26(128)
        self.mcse3 = nets.Block.MCSE26(256)
        self.mcse4 = nets.Block.MCSE26(512)

        self.final = nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x1 = nets.Block.cut(x)
        x = self.conv(x)
        x1 = self.conv(x1)

        out = self.layer1(x)
        out1 = self.layer_1(x1)
        mout1 = self.mcse1(out, out1)
        out = self.maxp(mout1)

        out = self.layer2(out)
        out2 = self.layer_2(self.maxp(out1))
        mout2 = self.mcse2(out, out2)
        out = self.maxp(mout2)

        out = self.layer3(out)
        out3 = self.layer_3(self.maxp(out2))
        mout3 = self.mcse3(out, out3)
        out = self.maxp(mout3)

        out = self.layer4(out)
        out4 = self.layer_4(self.maxp(out3))
        mout4 = self.mcse4(out, out4)
        out = self.maxp(mout4)

        out = self.layer5(out)
        output_4 = self.reblock4(out, mout4)
        out = self.upblock4(output_4, out)

        output_3 = self.reblock3(mout4, mout3)
        out = self.upblock3(output_3, out)

        output_2 = self.reblock2(mout3, mout2)
        out = self.upblock2(output_2, out)

        output_1 = self.reblock1(mout2, mout1)
        out = self.upblock1(output_1, out)

        out = self.final(out)

        return out

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.mynet.parameters():  # 将VGG16的特征提取层参数进行冻结，不对其进行更新
                param.requires_grad = False  # param.requires_grad = False:屏蔽预训练模型的权重，只训练全连接层的权重

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.mynet.parameters():
                param.requires_grad = True  # # param.requires_grad = true 所有层都参与训练