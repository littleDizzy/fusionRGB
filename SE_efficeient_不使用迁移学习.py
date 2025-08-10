import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

from torch import Tensor

import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
from torchvision.models import efficientnet_v2_s
import torch.nn.functional as F


# 将我们传入的channel的个数转换为距离8最近的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 卷积+BN+激活函数模块
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size: int = 3,  # 卷积核大小
                 stride: int = 1,
                 groups: int = 1,  # 用来控制我们深度可分离卷积的分组数(DWConv：这里要保证输入和输出的channel不会发生变化)
                 norm_layer: Optional[Callable[..., nn.Module]] = None,  # BN结构
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        # super() 函数接受两个参数：子类名和子类对象，用来指明在哪个子类中调用父类的方法。在这段代码中，ConvBNActivation 是子类名，self 是子类对象。
        # 通过super(ConvBNActivation, self)，Python知道要调用的是ConvBNActivation的父类的方法。
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=input_channel,
                                                         out_channels=output_channel,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(output_channel),
                                               activation_layer())


# SE模块：注意力机制
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_channel: int,  # block input channel
                 expand_channel: int,  # block expand channel 第一个1X1卷积扩展之后的channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_channel // squeeze_factor  # 第一个全连接层个数等于输入特征的1/4
        self.fc1 = nn.Conv2d(expand_channel, squeeze_c, 1)  # 压缩特征
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_channel, 1)  # 拓展特征
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 全局平均池化
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x  # 与输入的特征进行相乘


# 每个MBconv的配置参数
class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5 论文中的卷积核大小有3和5
                 input_channel: int,
                 out_channel: int,
                 expanded_ratio: int,  # 1 or 6 #第一个1x1卷积层的扩展倍数，论文中有1和6
                 stride: int,  # 1 or 2
                 use_se: bool,  # True 因为每个MBConv都使用SE模块 所以传入的参数是true
                 drop_rate: float,  # 随机失活比例
                 index: str,  # 1a, 2a, 2b, ... 用了记录当前MBconv当前的名称
                 width_coefficient: float):  # 网络宽度的倍率因子
        self.input_c = self.adjust_channels(input_channel, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_channel, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    # 后续如果想要继续使用B1~B7，可以使用B0的channel乘以倍率因子
    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


# 搭建MBconv模块
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)  # 当满足两个条件之后才能使用shortcut连接

        layers = OrderedDict()  # 创建空的有序字典用来保存MBConv
        activation_layer = nn.SiLU  # alias Swish

        # 搭建1x1升维卷积 ：这里其实是有个小技巧，论文中的MBconv的第一个1x1卷积是为了做升维操作，如果我们的expand为1的时候，可以不搭建第一个卷积层
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise 搭建深度可分离卷积（这里要保证输入和输出的channel不会发生变化）
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,  # 只有保证分组数和输入通道数保持一致才能确保输入和输入的channel保持不变
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})  # Identity 不做任何激活处理

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,  # 网络中最后一个全连接层的失活比例
                 drop_connect_rate: float = 0.2,  # 是MBconv中的随机失活率
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(input_channel=3,
                                                     output_channel=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(input_channel=last_conv_input_c,
                                               output_channel=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)

# ----------------------
# 模型训练与测试
# ----------------------
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device, far=0.001):
    """用GPU训练模型，并控制虚警率"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(num_epochs):
        # ----------------------
        # 训练阶段
        # ----------------------
        net.train()
        all_train_preds, all_train_labels = [], []
        train_loss = 0.0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_loss += l.item()

            # 收集训练集预测结果
            with torch.no_grad():
                preds = y_hat.argmax(dim=1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(y.cpu().numpy())

        # 计算训练集混淆矩阵
        train_cm = confusion_matrix(all_train_labels, all_train_preds)

        # ----------------------
        # 测试阶段（原始预测）
        # ----------------------
        net.eval()
        all_test_preds, all_test_labels = [], []
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                test_loss += l.item()
                preds = y_hat.argmax(dim=1)
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(y.cpu().numpy())

        # 原始测试混淆矩阵
        test_cm = confusion_matrix(all_test_labels, all_test_preds)

        # 打印训练结果
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_iter):.4f}")
        print(f"Test Loss: {test_loss / len(test_iter):.4f}")
        print("Train Confusion Matrix:")
        print(train_cm)
        print("Test Confusion Matrix (原始):")
        print(test_cm)

    # ----------------------
    # 虚警率控制核心逻辑
    # ----------------------
    # 收集测试集中所有负样本（标签0）的目标类概率
        net.eval()
        negative_probs = []
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                probs = torch.softmax(y_hat, dim=1)[:, 1]  # 目标类（第1类）的概率
                mask = (y == 0)  # 仅保留负样本
                negative_probs.extend(probs[mask].cpu().numpy())

        # 计算阈值（99.9%分位数）
        threshold = np.percentile(negative_probs, 100 * (1 - far))
        print(f"\n[虚警控制] 阈值 (FAR={far}): {threshold:.4f}")

        # 用阈值重新评估测试集
        all_test_preds, all_test_labels = [], []
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                probs = torch.softmax(y_hat, dim=1)[:, 1]
                preds = (probs > threshold).int().cpu().numpy()  # 应用阈值
                all_test_preds.extend(preds)
                all_test_labels.extend(y.cpu().numpy())

        # 最终混淆矩阵
        final_test_cm = confusion_matrix(all_test_labels, all_test_preds)
        print("\n[虚警控制] 最终测试混淆矩阵:")
        print(final_test_cm)

        # 计算性能指标
        tn, fp, fn, tp = final_test_cm.ravel()
        far_actual = fp / (fp + tn)
        recall = tp / (tp + fn)
        print(f"实际虚警率: {far_actual:.4f}")
        print(f"召回率: {recall:.4f}")

        # 计算训练速度
        total_time = time.time() - start_time
        num_examples = sum(y.shape[0] for X, y in train_iter) * num_epochs
        print(f'\n训练速度: {num_examples / total_time:.1f} examples/sec on {str(device)}')

    return net
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr, num_epochs, batch_size, far = 0.01, 50, 64, 0.001
    data_dir = r"D:\time2image\data17\NTFD\fusionRGB"

    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3561, 0.3561, 0.3561], std=[0.1262, 0.1262, 0.1262])
    ])

    dataset = ImageFolder(root=data_dir, transform=transformer)
    print(dataset.class_to_idx)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    feature, label = next(iter(train_iter))
    print(feature.shape, label)

    # 读取模型
    model = efficientnet_b0(num_classes=2).to(device)
    # 修改分类头（保持原始特征提取层不变）
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)  # 输出2类
    )
    # 打印模型结构
    print(model)
    train_ch6(model, train_iter, test_iter, num_epochs=num_epochs, lr=lr, device=device, far=far)
