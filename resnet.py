'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from misc import model_utils
from torch.nn.parameter import Parameter


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class HW_connection(nn.Module):
    def __init__(self, planes, subtract_mean=False, trans_gate_bias=0, carry_gate_bias=0, normal=False, skip_sum_1=False, nonlinear=nn.Sigmoid()):
        super(HW_connection, self).__init__()
        self.normal = normal
        self.skip_sum_1 = skip_sum_1
        self.nonlinear = nonlinear
        self.subtract_mean = subtract_mean
        if self.subtract_mean:
            self.bn = nn.BatchNorm2d(planes, affine=False)

        self.transform_gate = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
                                            self.nonlinear)
        self.transform_gate[0].bias.data.fill_(trans_gate_bias)
        if not self.skip_sum_1:
            self.carry_gate = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
                self.nonlinear)
            self.carry_gate[0].bias.data.fill_(carry_gate_bias)

    def forward(self, input_1, input_2):
        # both inputs' size maybe batch*planes*H*W
        trans_gate = self.transform_gate(input_1)   # batch*planes*H*W
        if self.skip_sum_1:
            carry_gate = 1 - trans_gate
        else:
            carry_gate = self.carry_gate(input_1)       # batch*planes*H*W

        if self.subtract_mean:
            output = input_2 * trans_gate + input_1 * carry_gate
            self.bn(output)
            running_mean = self.bn.running_mean.unsqueeze(dim=1).unsqueeze(dim=2)
            output = output - running_mean
            if self.normal == True:
                l2 = torch.stack([trans_gate, carry_gate], dim=4).norm(p=2, dim=4, keepdim=False)
                output = output/l2
            output += running_mean
        else:
            if self.normal == True:
                l2 = torch.stack([trans_gate, carry_gate], dim=4).norm(p=2, dim=4, keepdim=False)
                trans_gate = trans_gate/l2
                carry_gate = carry_gate/l2
            output = input_2 * trans_gate + input_1 * carry_gate  # batch*opt.rnn_size

        trans_gate = trans_gate.mean()
        carry_gate = carry_gate.mean()

        return output, trans_gate, carry_gate


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, opt, skip, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if opt.pre_activation:
            self.bn1 = nn.BatchNorm2d(in_planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

        self.pre_activation = opt.pre_activation
        self.skip = skip
        if skip == 'HW':
            self.HW_connection = HW_connection(planes=planes, trans_gate_bias=opt.skip_HW_trans_bias, carry_gate_bias=opt.skip_HW_carry_bias, normal=False)
        elif skip == 'HW-normal':
            self.HW_connection = HW_connection(planes=planes, trans_gate_bias=opt.skip_HW_trans_bias, carry_gate_bias=opt.skip_HW_carry_bias, normal=True)
        elif skip == 'HW-normal-sub':
            self.HW_connection = HW_connection(planes=planes, subtract_mean=True, trans_gate_bias=opt.skip_HW_trans_bias,
                                               carry_gate_bias=opt.skip_HW_carry_bias, normal=True)

    def forward(self, x):
        if self.pre_activation:
            out = F.relu(self.bn1(x))
            out = F.relu(self.bn2(self.conv1(out)))
            out = self.conv2(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

        input = self.shortcut(x)
        if self.skip == 'RES':
            out += input
        elif self.skip == 'RES-l2':
            out += input
            out = out * (1.0 / (2 ** 0.5))
        elif self.skip is not None and self.skip in ['HW', 'HW-normal', 'HW-normal-sub']:
            out, trans_gate, carry_gate = self.HW_connection(input, out)
            # print('carry_gate and trans_gate:', carry_gate.item(), trans_gate.item())

        if not self.pre_activation:
            out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, opt, num_classes=10):
        super(ResNet, self).__init__()
        self.opt = opt
        self.in_planes = 16

        # self.skip_connection = opt.skip_connection
        self.skip_2_num = opt.skip_2_num
        num_skip_2 = self._get_num_skip_2(self.skip_2_num, num_blocks)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], num_skip_2[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], num_skip_2[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], num_skip_2[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        if opt.cappro:
            del self.linear
            self.linear = LinearCapsPro(opt, 64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, num_skip_2, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(0, len(strides)):
            if i < num_blocks-num_skip_2:
                layers.append(block(self.in_planes, planes, self.opt, skip=self.opt.skip_connection_1, stride=strides[i]))
            else:
                layers.append(block(self.in_planes, planes, self.opt, skip=self.opt.skip_connection_2, stride=strides[i]))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _get_num_skip_2(self, skip_2_num, num_blocks):

        if skip_2_num <= num_blocks[2]:
            return [0, 0, skip_2_num]
        elif skip_2_num <= num_blocks[1] + num_blocks[2]:
            return [0, skip_2_num - num_blocks[2], num_blocks[2]]
        elif skip_2_num <= num_blocks[0] + num_blocks[1] + num_blocks[2]:
            return [skip_2_num - num_blocks[2] - num_blocks[1], num_blocks[1], num_blocks[2]]
        else:
            return num_blocks

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(opt):
    return ResNet(BasicBlock, [3, 3, 3], opt)


def resnet32(opt):
    return ResNet(BasicBlock, [5, 5, 5], opt)


def resnet44(opt):
    return ResNet(BasicBlock, [7, 7, 7], opt)


def resnet56(opt):
    return ResNet(BasicBlock, [9, 9, 9], opt)


def resnet110(opt):
    return ResNet(BasicBlock, [18, 18, 18], opt)


def resnet1202(opt):
    return ResNet(BasicBlock, [200, 200, 200], opt)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


class LinearCapsPro(nn.Module):
    def __init__(self, opt, in_features, num_C):
        super(LinearCapsPro, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.weight = Parameter(torch.Tensor(self.num_C, in_features))

        self.opt = opt
        self.init = opt.cappro_init_method
        self.cappro_pro_mul_w = opt.cappro_pro_mul_w
        self.cappro_dis_mul_w = opt.cappro_dis_mul_w
        self.cappro_method = opt.cappro_method

        self.reset_parameters()

    def reset_parameters(self):
        if self.init == 'kaiming':
            nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.weight)
        elif self.init == 'orthogonal':
            nn.init.orthogonal_(self.weight)

    def forward(self, x):
        wx = torch.matmul(x, torch.t(self.weight))          # batch*num_classes
        w_len_pow2 = torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))  # 1*num_classes
        if self.cappro_method in ['pro', 'pro-dis']:
            pro = wx                                                # batch*num_classes
            if not self.cappro_pro_mul_w:
                pro = pro / torch.sqrt(w_len_pow2)
        else:
            pro = 0

        if self.cappro_method in ['dis', 'pro-dis']:
            x_len_pow2 = x.pow(2).sum(dim=1, keepdim=True)      # batch*1
            wx_pow2 = wx.pow(2)     # batch*num_classes

            if self.training:
                x_len_pow2 = x_len_pow2 * (1.0 - self.opt.drop_prob_output)

            dis = torch.sqrt(F.relu(x_len_pow2 - wx_pow2 / w_len_pow2))         # batch*num_classes
            dis = torch.sign(wx)*(dis - torch.sqrt(x_len_pow2))                 # batch*num_classes

            if self.cappro_dis_mul_w:
                dis = dis * torch.sqrt(w_len_pow2)

        else:
            dis = 0

        return pro-dis



if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()