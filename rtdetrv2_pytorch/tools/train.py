"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# 导入所需的系统模块
import os 
import sys 
# 将当前文件的上级目录添加到系统路径,以便导入自定义模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

# 导入自定义模块
from src.misc import dist_utils  # 分布式训练相关工具
from src.core import YAMLConfig, yaml_utils  # YAML配置文件处理工具
from src.solver import TASKS  # 任务解决器字典


def main(args, ) -> None:
    """主函数
    Args:
        args: 命令行参数
    """
    # 设置分布式训练环境
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # 确保不会同时进行微调和恢复训练
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # 解析命令行更新参数并与其他参数合并
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    # 创建配置对象并打印配置信息
    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    # 根据任务类型创建对应的solver实例
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    # 如果是测试模式则只进行验证,否则进行训练
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    # 清理分布式训练环境
    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # 优先级0的参数
    parser.add_argument('-c', '--config', type=str, required=True)  # 配置文件路径,必需参数
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')  # 从检查点恢复训练
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')  # 从检查点开始微调
    parser.add_argument('-d', '--device', type=str, help='device',)  # 指定设备
    parser.add_argument('--seed', type=int, help='exp reproducibility')  # 随机种子,用于实验可重复性
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')  # 是否使用混合精度训练
    parser.add_argument('--output-dir', type=str, help='output directoy')  # 输出目录
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')  # tensorboard日志目录
    parser.add_argument('--test-only', action='store_true', default=False,)  # 是否只进行测试

    # 优先级1的参数
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')  # YAML配置文件的更新参数

    # 环境相关参数
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')  # 打印方法
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')  # 打印进程的rank id

    parser.add_argument('--local-rank', type=int, help='local rank id')  # 本地进程的rank id
    args = parser.parse_args()

    main(args)
