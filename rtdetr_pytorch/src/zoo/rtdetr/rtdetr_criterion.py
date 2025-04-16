"""
参考实现:
https://github.com/facebookresearch/detr/blob/main/models/detr.py

作者: lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

# 从box_ops导入边界框转换和IoU计算相关函数
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

# 导入分布式训练相关函数
from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register


@register
class SetCriterion(nn.Module):
    """DETR的损失函数计算类
    
    损失计算分两步:
    1) 使用匈牙利算法计算预测框和真实框之间的匹配
    2) 对每对匹配的预测框和真实框计算监督损失(包括类别损失和边界框损失)
    """
    __share__ = ['num_classes', ]  # 共享参数
    __inject__ = ['matcher', ]     # 需要注入的组件

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """初始化损失函数计算类
        
        Args:
            matcher: 用于计算预测框和真实框匹配的模块
            weight_dict: 字典,包含各个损失的权重
            losses: 需要计算的损失列表
            alpha: Focal Loss的alpha参数,默认0.2
            gamma: Focal Loss的gamma参数,默认2.0  
            eos_coef: 背景类别的相对权重,默认1e-4
            num_classes: 目标类别数,不包括背景类,默认80
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        # 创建类别权重向量,背景类权重为eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """分类损失计算(使用交叉熵)
        
        Args:
            outputs: 模型输出字典,必须包含'pred_logits'
            targets: 目标字典列表,每个字典必须包含'labels'
            indices: 预测框和真实框的匹配索引
            num_boxes: 标准化系数
            log: 是否记录分类准确率,默认True
            
        Returns:
            losses: 包含分类损失的字典
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # 记录分类错误率
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        """二值交叉熵分类损失
        
        Args:
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引
            num_boxes: 标准化系数
            log: 是否记录日志
            
        Returns:
            包含BCE损失的字典
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Focal Loss分类损失
        
        使用Focal Loss处理类别不平衡问题
        
        Args:
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引  
            num_boxes: 标准化系数
            log: 是否记录日志
            
        Returns:
            包含focal loss的字典
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        """Varifocal Loss分类损失
        
        结合IoU分数的Focal Loss变体
        
        Args:
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引
            num_boxes: 标准化系数
            log: 是否记录日志
            
        Returns:
            包含varifocal loss的字典
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # 计算预测框和真实框的IoU
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        # 准备分类目标
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # 使用IoU作为目标分数
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        # 计算权重
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        # 计算加权BCE损失
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """计算基数误差
        
        基数误差是预测的非空框数量与真实框数量之间的绝对误差
        这不是真正的损失,仅用于日志记录,不传递梯度
        
        Args:
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引
            num_boxes: 标准化系数
            
        Returns:
            包含基数误差的字典
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # 统计预测为非背景类的数量
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算边界框相关的损失
        
        包括L1回归损失和GIoU损失
        目标框格式为(center_x, center_y, w, h),经过图像大小归一化
        
        Args:
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引
            num_boxes: 标准化系数
            
        Returns:
            包含边界框L1损失和GIoU损失的字典
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        # L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # GIoU损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """计算掩码相关的损失
        
        包括focal loss和dice loss
        目标字典必须包含'masks'键,对应的张量维度为[nb_target_boxes, h, w]
        
        Args:
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引
            num_boxes: 标准化系数
            
        Returns:
            包含掩码focal loss和dice loss的字典
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # 上采样预测到目标大小
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        """获取源(预测)的置换索引
        
        Args:
            indices: 匹配索引列表
            
        Returns:
            batch_idx: 批次索引
            src_idx: 源索引
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """获取目标的置换索引
        
        Args:
            indices: 匹配索引列表
            
        Returns:
            batch_idx: 批次索引
            tgt_idx: 目标索引
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """获取指定类型的损失
        
        Args:
            loss: 损失类型名称
            outputs: 模型输出
            targets: 目标
            indices: 匹配索引
            num_boxes: 标准化系数
            **kwargs: 额外参数
            
        Returns:
            计算得到的损失
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """前向传播函数
        
        计算所有损失并返回
        
        Args:
            outputs: 模型输出字典
            targets: 目标字典列表,batch_size个字典
            
        Returns:
            losses: 包含所有损失的字典
        """
        # 移除辅助输出
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # 计算最后一层输出和目标之间的匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算目标框的平均数量(用于归一化)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # 处理辅助输出的损失
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # 中间层的mask损失计算成本太高,忽略
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # 只在最后一层启用日志记录
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 处理CDN(Contrastive Denoising)辅助损失
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """获取CDN匹配的索引
        
        Args:
            dn_meta: CDN元信息
            targets: 目标列表
            
        Returns:
            dn_match_indices: CDN匹配索引列表
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """计算top-k准确率
    
    Args:
        output: 模型输出
        target: 目标标签
        topk: 要计算的k值元组,默认(1,)
        
    Returns:
        res: top-k准确率列表
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

