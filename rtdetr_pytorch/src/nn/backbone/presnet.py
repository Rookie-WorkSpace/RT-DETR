'''
作者: lyuwenyu
实现了PResNet主干网络,支持18/34/50/101层配置
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = ['PResNet']


# 不同深度ResNet的每个stage的block数量配置
ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


# 预训练模型下载地址
donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class BasicBlock(nn.Module):
    """基本残差块,用于ResNet18/34
    
    Args:
        ch_in (int): 输入通道数
        ch_out (int): 输出通道数
        stride (int): 步长
        shortcut (bool): 是否使用shortcut连接
        act (str): 激活函数类型
        variant (str): 网络变体类型,可选'b'或'd'
    """
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                # d变体使用avgpool+1x1conv作为shortcut
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        # 主分支包含两个3x3卷积
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    """瓶颈残差块,用于ResNet50/101/152
    
    Args:
        ch_in (int): 输入通道数
        ch_out (int): 输出通道数
        stride (int): 步长
        shortcut (bool): 是否使用shortcut连接
        act (str): 激活函数类型
        variant (str): 网络变体类型,可选'a','b'或'd'
    """
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            # a变体将stride应用在第一个1x1卷积
            stride1, stride2 = stride, 1
        else:
            # b/d变体将stride应用在3x3卷积
            stride1, stride2 = 1, stride

        width = ch_out 

        # 主分支包含1x1->3x3->1x1三个卷积
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                # d变体使用avgpool+1x1conv作为shortcut
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    """构建ResNet的一个stage,包含多个残差块
    
    Args:
        block: 残差块类型(BasicBlock或BottleNeck)
        ch_in (int): 输入通道数
        ch_out (int): 输出通道数
        count (int): 残差块数量
        stage_num (int): stage序号
        act (str): 激活函数类型
        variant (str): 网络变体类型
    """
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register
class PResNet(nn.Module):
    """PResNet主干网络
    
    Args:
        depth (int): 网络深度,支持18/34/50/101
        variant (str): 网络变体类型,默认'd'
        num_stages (int): stage数量,默认4
        return_idx (list): 输出的stage索引
        act (str): 激活函数类型
        freeze_at (int): 冻结的stage数量
        freeze_norm (bool): 是否冻结归一化层
        pretrained (bool): 是否加载预训练权重
    """
    def __init__(
        self, 
        depth, 
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]  # 获取每个stage的block数量
        ch_in = 64  # 输入通道数
        if variant in ['c', 'd']:
            # c/d变体使用3个3x3卷积替代7x7卷积
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],  # 第一个3x3卷积
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],  # 第二个3x3卷积
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],  # 第三个3x3卷积
            ]
        else:
            # 标准版使用7x7卷积
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        # 构建stem层
        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]  # 每个stage的输出通道数
        block = BottleNeck if depth >= 50 else BasicBlock  # 选择BottleNeck或BasicBlock作为残差块

        _out_channels = [block.expansion * v for v in ch_out_list]  # 每个stage的输出通道数
        _out_strides = [4, 8, 16, 32]  # 每个stage的输出步长

        # 构建4个stage
        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        # 冻结指定数量的stage
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        # 冻结归一化层
        if freeze_norm:
            self._freeze_norm(self)

        # 加载预训练权重
        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')
            
    def _freeze_parameters(self, m: nn.Module):
        """冻结模块参数"""
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        """冻结归一化层"""
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            list: 返回多个stage的特征图
        """
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


