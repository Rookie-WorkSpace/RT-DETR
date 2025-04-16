"""RT-DETR模型实现
作者: lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    """RT-DETR模型类
    
    实现了RT-DETR (Real-Time DEtection TRansformer)模型的主体架构,包含backbone、encoder和decoder三个主要组件。
    支持多尺度训练和部署模式转换。

    Attributes:
        __inject__: 需要注入的组件列表,包括backbone、encoder和decoder
        backbone: 主干网络,用于提取图像特征
        encoder: 编码器,处理backbone输出的特征
        decoder: 解码器,生成最终的检测结果
        multi_scale: 多尺度训练的尺度列表,如果为None则不使用多尺度训练
    """
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        """初始化RT-DETR模型
        
        Args:
            backbone (nn.Module): 主干网络模型
            encoder: 编码器模型
            decoder: 解码器模型
            multi_scale (list, optional): 多尺度训练的尺度列表. Defaults to None.
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        """前向传播函数
        
        Args:
            x (Tensor): 输入图像张量
            targets (dict, optional): 训练时的标签信息. Defaults to None.
            
        Returns:
            dict: 模型输出结果
        """
        # 训练时如果启用多尺度,随机选择一个尺度并调整图像大小
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale) # 随机选择一个尺度  
            x = F.interpolate(x, size=[sz, sz]) # 调整图像大小
            
        # 依次通过backbone、encoder和decoder处理
        x = self.backbone(x)    # 提取特征
        x = self.encoder(x)     # 编码特征
        x = self.decoder(x, targets)  # 解码特征

        return x
    
    def deploy(self, ):
        """转换模型到部署模式
        
        将模型设置为评估模式,并对所有支持部署转换的模块进行转换。
        
        Returns:
            RTDETR: 返回转换后的模型自身
        """
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
