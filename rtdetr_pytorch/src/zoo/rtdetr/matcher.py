"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
用于计算匹配代价并求解相应的线性分配问题的模块。

作者: lyuwenyu
"""

import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from src.core import register


@register
class HungarianMatcher(nn.Module):
    """该类用于计算网络预测结果和目标之间的匹配关系

    出于效率考虑,目标不包含背景类。因此通常预测数量会多于目标数量。
    在这种情况下,我们对最佳预测进行一对一匹配,而其他预测则视为未匹配(即被视为背景)。
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """初始化匹配器

        参数:
            cost_class: 分类误差在匹配代价中的相对权重
            cost_bbox: 边界框坐标L1误差在匹配代价中的相对权重
            cost_giou: 边界框GIoU损失在匹配代价中的相对权重
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """执行匹配操作

        参数:
            outputs: 包含以下条目的字典:
                 "pred_logits": 形状为[batch_size, num_queries, num_classes]的张量,包含分类logits
                 "pred_boxes": 形状为[batch_size, num_queries, 4]的张量,包含预测的边界框坐标

            targets: 目标列表(长度为batch_size),每个目标是包含以下内容的字典:
                 "labels": 形状为[num_target_boxes]的张量(其中num_target_boxes是目标中真实对象的数量),包含类别标签
                 "boxes": 形状为[num_target_boxes, 4]的张量,包含目标边界框坐标

        返回:
            大小为batch_size的列表,包含(index_i, index_j)元组,其中:
                - index_i是选定预测的索引(按顺序)
                - index_j是对应选定目标的索引(按顺序)
            对于每个batch元素,满足:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 展平以批量计算代价矩阵
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 连接目标标签和边界框
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类代价。与损失不同,我们不使用NLL,
        # 而是用1 - proba[target class]近似。
        # 这里的1是一个不影响匹配的常数,可以省略。
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # 计算边界框之间的L1代价
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算边界框之间的giou代价
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # 最终代价矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
