"""
参考来源:
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

作者: lyuwenyu
"""

import random
import numpy as np 

import torch
import torch.nn as nn 
import torch.distributed
import torch.distributed as tdist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def init_distributed():
    '''
    初始化分布式训练环境
    参数:
        backend (str): 后端类型,可选 'nccl' 或 'gloo'
    返回:
        bool: 是否成功初始化
    '''
    try:
        # # PyTorch分布式训练环境变量
        # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 本地进程序号
        # RANK = int(os.getenv('RANK', -1))             # 全局进程序号
        # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 总进程数
        
        # 初始化进程组
        tdist.init_process_group(init_method='env://', )
        torch.distributed.barrier()

        # 获取当前进程序号并设置对应GPU设备
        rank = get_rank()
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        # 设置只在主进程打印信息
        setup_print(rank == 0)
        print('Initialized distributed mode...')

        return True 

    except:
        print('Not init distributed mode.')
        return False 


def setup_print(is_main):
    '''
    设置打印函数,使得只在主进程中打印信息
    参数:
        is_main (bool): 是否为主进程
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    """
    检查分布式环境是否可用且已初始化
    返回:
        bool: 分布式环境状态
    """
    if not tdist.is_available():
        return False
    if not tdist.is_initialized():
        return False
    return True


def get_rank():
    """
    获取当前进程的序号
    返回:
        int: 进程序号,非分布式返回0
    """
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_world_size():
    """
    获取总进程数
    返回:
        int: 总进程数,非分布式返回1
    """
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()

    
def is_main_process():
    """
    判断当前是否为主进程
    返回:
        bool: 是否为主进程
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    仅在主进程上保存模型
    参数:
        *args: 传递给torch.save的位置参数
        **kwargs: 传递给torch.save的关键字参数
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def warp_model(model, find_unused_parameters=False, sync_bn=False,):
    """
    封装模型以用于分布式训练
    参数:
        model: PyTorch模型
        find_unused_parameters (bool): 是否查找未使用的参数
        sync_bn (bool): 是否使用同步批归一化
    返回:
        封装后的模型
    """
    if is_dist_available_and_initialized(): # 如果分布式环境可用且已初始化
        rank = get_rank()   # 获取当前进程的序号
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model    # 同步批归一化
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)    # 分布式训练
    return model


def warp_loader(loader, shuffle=False):        
    """
    封装数据加载器以用于分布式训练
    参数:
        loader: PyTorch数据加载器
        shuffle (bool): 是否打乱数据
    返回:
        封装后的数据加载器
    """
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset, 
                            loader.batch_size, 
                            sampler=sampler, 
                            drop_last=loader.drop_last, 
                            collate_fn=loader.collate_fn, 
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers, )
    return loader


def is_parallel(model) -> bool:
    """
    判断模型是否为并行模型(DP或DDP)
    参数:
        model: PyTorch模型
    返回:
        bool: 是否为并行模型
    """
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    """
    移除模型的并行包装
    参数:
        model: 并行模型
    返回:
        nn.Module: 去除并行包装后的模型
    """
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    '''
    规约字典数据,用于多进程间的数据聚合
    参数:
        data (dict): 输入字典
        avg (bool): 是否对结果取平均
    返回:
        dict: 规约后的字典
    '''
    world_size = get_world_size()
    if world_size < 2:
        return data
    
    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size
        
        _data = {k: v for k, v in zip(keys, values)}
    
    return _data


def all_gather(data):
    """
    收集所有进程的数据
    参数:
        data: 任意可序列化的数据对象
    返回:
        list: 包含所有进程数据的列表
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list

    
import time 
def sync_time():
    '''
    同步时间,确保所有GPU操作完成后再记录时间
    返回:
        float: 当前时间戳
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def set_seed(seed):
    """
    设置随机种子以确保实验可重复性
    参数:
        seed (int): 随机种子
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

