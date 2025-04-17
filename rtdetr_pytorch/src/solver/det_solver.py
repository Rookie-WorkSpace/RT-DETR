'''
作者: lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        """
        训练主函数
        """
        print("start training")
        self.train()

        args = self.cfg 
        
        # 计算模型参数量
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model parameters: {n_parameters}')

        # 获取验证集的COCO API接口
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # 记录最佳模型状态
        best_stat = {'epoch': -1, }

        start_time = time.time()
        # 开始训练循环
        print(f'Total epochs: {args.epoches}')
        for epoch in range(self.last_epoch + 1, args.epoches):
            # 如果是分布式训练,设置采样器epoch
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # 训练一个epoch
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            # 学习率调整
            self.lr_scheduler.step()
            
            # 保存检查点
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # 每checkpoint_step个epoch保存一次
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            # 验证
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # 更新最佳模型状态
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('最佳状态: ', best_stat)

            # 记录训练和测试的统计信息
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            # 保存日志
            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # 保存评估日志
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        # 打印总训练时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('训练总时长 {}'.format(total_time_str))


    def val(self, ):
        """
        验证函数
        """
        self.eval()

        # 获取验证集的COCO API接口
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        # 使用EMA模型或原始模型进行验证
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        # 保存评估结果
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
