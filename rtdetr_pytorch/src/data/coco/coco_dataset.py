"""
COCO数据集实现,返回image_id用于评估。
主要代码来自 https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision
# 禁用beta transforms警告
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register

__all__ = ['CocoDetection']


@register
class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO检测数据集类
    继承自torchvision的CocoDetection类
    """
    __inject__ = ['transforms']  # 注入transforms参数
    __share__ = ['remap_mscoco_category']  # 共享remap_mscoco_category参数
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        """初始化函数
        Args:
            img_folder: 图像文件夹路径
            ann_file: 标注文件路径
            transforms: 数据增强转换
            return_masks: 是否返回实例分割掩码
            remap_mscoco_category: 是否重映射MSCOCO类别ID
        """
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        """获取单个数据样本
        Args:
            idx: 索引
        Returns:
            img: 图像
            target: 标注信息
        """
        import PIL  
        PIL.Image.MAX_IMAGE_PIXELS = None 
        
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # 转换边界框和掩码格式
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target

    def extra_repr(self) -> str:
        """返回额外的字符串表示
        Returns:
            str: 数据集配置的字符串表示
        """
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 


def convert_coco_poly_to_mask(segmentations, height, width):
    """将COCO多边形标注转换为二值掩码
    Args:
        segmentations: 分割标注
        height: 图像高度
        width: 图像宽度
    Returns:
        masks: 二值掩码张量
    """
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    """转换COCO标注格式的类"""
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        """初始化函数
        Args:
            return_masks: 是否返回实例分割掩码
            remap_mscoco_category: 是否重映射MSCOCO类别ID
        """
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        """转换单个样本的标注格式
        Args:
            image: PIL图像
            target: 原始标注字典
        Returns:
            image: 原始图像
            target: 转换后的标注字典
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        # 过滤掉crowd实例
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 提取并处理边界框
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # 转换为xyxy格式
        boxes[:, 0::2].clamp_(min=0, max=w)  # 裁剪到图像范围内
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 处理类别ID
        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        # 处理实例分割掩码
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # 处理关键点(如果有)
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # 过滤无效边界框
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # 构建目标字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # 添加COCO API所需的额外信息
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # 记录图像尺寸信息
        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


# MSCOCO类别ID到类别名称的映射
# mscoco_category2name = {
#     1: 'person',
#     2: 'bicycle',
#     3: 'car',
#     4: 'motorcycle',
#     5: 'airplane',
#     6: 'bus',
#     7: 'train',
#     8: 'truck',
#     9: 'boat',
#     10: 'traffic light',
#     11: 'fire hydrant',
#     13: 'stop sign',
#     14: 'parking meter',
#     15: 'bench',
#     16: 'bird',
#     17: 'cat',
#     18: 'dog',
#     19: 'horse',
#     20: 'sheep',
#     21: 'cow',
#     22: 'elephant',
#     23: 'bear',
#     24: 'zebra',
#     25: 'giraffe',
#     27: 'backpack',
#     28: 'umbrella',
#     31: 'handbag',
#     32: 'tie',
#     33: 'suitcase',
#     34: 'frisbee',
#     35: 'skis',
#     36: 'snowboard',
#     37: 'sports ball',
#     38: 'kite',
#     39: 'baseball bat',
#     40: 'baseball glove',
#     41: 'skateboard',
#     42: 'surfboard',
#     43: 'tennis racket',
#     44: 'bottle',
#     46: 'wine glass',
#     47: 'cup',
#     48: 'fork',
#     49: 'knife',
#     50: 'spoon',
#     51: 'bowl',
#     52: 'banana',
#     53: 'apple',
#     54: 'sandwich',
#     55: 'orange',
#     56: 'broccoli',
#     57: 'carrot',
#     58: 'hot dog',
#     59: 'pizza',
#     60: 'donut',
#     61: 'cake',
#     62: 'chair',
#     63: 'couch',
#     64: 'potted plant',
#     65: 'bed',
#     67: 'dining table',
#     70: 'toilet',
#     72: 'tv',
#     73: 'laptop',
#     74: 'mouse',
#     75: 'remote',
#     76: 'keyboard',
#     77: 'cell phone',
#     78: 'microwave',
#     79: 'oven',
#     80: 'toaster',
#     81: 'sink',
#     82: 'refrigerator',
#     84: 'book',
#     85: 'clock',
#     86: 'vase',
#     87: 'scissors',
#     88: 'teddy bear',
#     89: 'hair drier',
#     90: 'toothbrush'
# }

# 成都数据集的类别映射
mscoco_category2name = {
    0: '划痕',
    1: '吊紧',
    2: '拼接间隙',
    3: '水渍',
    4: '水珠',
    5: '爆线',
    6: '破损',
    7: '碰伤',
    8: '红标签',
    9: '线头',
    10: '织物外漏',
    11: '缝线鼓包(轻度)',
    12: '脏污',
    13: '褶皱（轻度）',
    14: '褶皱（重度）',
    15: '跳针',
    16: '针眼'
}

# 类别ID与标签的相互映射
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}