"""
作者: lyuwenyu
"""

import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
from typing import Dict

from src.misc import dist
from src.core import BaseConfig


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        """初始化求解器
        Args:
            cfg: 配置对象,包含模型训练所需的各种参数设置
        """
        self.cfg = cfg 

    def setup(self, ):
        """设置训练环境,避免不必要的类实例化
        - 设置设备(CPU/GPU)
        - 加载模型并进行分布式封装
        - 加载损失函数和后处理器
        - 设置混合精度训练
        - 设置EMA
        - 创建输出目录
        """
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        # 封装模型用于分布式训练
        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        # 注意:在构建EMA实例之前需要加载微调状态
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        # 设置混合精度训练和EMA
        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 

        # 创建输出目录
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def train(self, ):
        """准备训练
        - 调用setup()进行基础设置
        - 设置优化器和学习率调度器
        - 如果需要则恢复checkpoint
        - 准备训练和验证数据加载器
        """
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # 注意实例化顺序
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        # 封装数据加载器用于分布式训练
        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)


    def eval(self, ):
        """准备评估
        - 调用setup()进行基础设置
        - 准备验证数据加载器
        - 如果需要则恢复checkpoint
        """
        self.setup()
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)


    def state_dict(self, last_epoch):
        """保存模型训练状态
        Args:
            last_epoch: 最后一个epoch的索引
        Returns:
            state: 包含模型参数、优化器状态等的字典
        """
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO: 待完善
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state


    def load_state_dict(self, state):
        """加载模型训练状态
        Args:
            state: 包含模型参数、优化器状态等的字典
        """
        # TODO: 待完善
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')


    def save(self, path):
        """保存模型状态到文件
        Args:
            path: 保存路径
        """
        state = self.state_dict()
        dist.save_on_master(state, path)


    def resume(self, path):
        """从文件恢复模型状态
        Args:
            path: checkpoint文件路径
        """
        # 加载到CPU内存以节省CUDA内存
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path,):
        """仅加载模型参数用于微调,跳过缺失/不匹配的键
        Args:
            path: 预训练模型路径或URL
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)
        
        # TODO: 硬编码处理
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        """匹配两个状态字典中的参数
        Args:
            state: 当前模型的状态字典
            params: 加载的参数字典
        Returns:
            matched_state: 匹配的参数字典
            infos: 包含未匹配和形状不匹配参数信息的字典
        """
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        """训练入口函数,需要在子类中实现"""
        raise NotImplementedError('')

    def val(self, ):
        """验证入口函数,需要在子类中实现"""
        raise NotImplementedError('')
