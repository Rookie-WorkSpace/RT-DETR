"""
RT-DETR后处理模块
by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    """RT-DETR后处理器,用于处理模型输出的预测结果
    
    Args:
        num_classes (int): 类别数量,默认80
        use_focal_loss (bool): 是否使用focal loss,默认True
        num_top_queries (int): 保留的top-k预测框数量,默认300
        remap_mscoco_category (bool): 是否重映射MSCOCO类别ID,默认False
    """
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        """返回额外的字符串表示"""
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def forward(self, outputs, orig_target_sizes):
        """前向推理函数
        
        Args:
            outputs (dict): 模型输出字典,包含pred_logits和pred_boxes
            orig_target_sizes (Tensor): 原始图像尺寸
            
        Returns:
            deploy模式: (labels, boxes, scores)元组
            训练模式: 包含labels、boxes、scores的字典列表
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # 将预测框从cxcywh格式转换为xyxy格式,并缩放到原图尺寸
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            # focal loss模式:对预测分数进行sigmoid,取top-k
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            # 非focal loss模式:对预测分数进行softmax,取最大值
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # ONNX导出模式
        if self.deploy_mode:
            return labels, boxes, scores

        # 重映射MSCOCO类别ID
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        # 组装结果
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        
    def deploy(self, ):
        """切换到部署模式"""
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        """返回IoU计算类型"""
        return ('bbox', )
