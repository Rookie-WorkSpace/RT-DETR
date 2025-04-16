"""
训练脚本入口
作者: lyuwenyu
"""

# 导入必要的库
import os 
import sys 
# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

# 导入自定义模块
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''
    主函数:
    1. 初始化分布式训练环境
    2. 设置随机种子
    3. 加载配置文件
    4. 创建训练器
    5. 执行训练或验证
    '''
    # 初始化分布式训练
    dist.init_distributed()
    # 如果指定了随机种子则设置
    if args.seed is not None:
        dist.set_seed(args.seed)

    # 确保不同时使用tuning和resume模式
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # 创建配置对象
    cfg = YAMLConfig(
        args.config,      # 配置文件路径
        resume=args.resume,   # 是否从检查点恢复训练
        use_amp=args.amp,     # 是否使用混合精度训练
        tuning=args.tuning    # 是否使用微调模式
    )

    
    # 根据配置创建对应任务的训练器
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    # 如果是测试模式则只执行验证,否则执行完整训练
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--config', '-c', type=str, )  # 配置文件路径
    parser.add_argument('--resume', '-r', type=str, )  # 恢复训练的检查点路径
    parser.add_argument('--tuning', '-t', type=str, )  # 微调模式的预训练模型路径
    parser.add_argument('--test-only', action='store_true', default=False,)  # 是否只执行测试
    parser.add_argument('--amp', action='store_true', default=False,)  # 是否启用自动混合精度
    parser.add_argument('--seed', type=int, help='seed',)  # 随机种子
    args = parser.parse_args()

    main(args)
