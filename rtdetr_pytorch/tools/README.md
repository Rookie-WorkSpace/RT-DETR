

训练/测试脚本示例
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint` # 从检查点恢复训练
- `--amp` # 使用混合精度训练
- `--test-only` # 仅执行测试


微调脚本示例
- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth` 


导出模型脚本示例
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


GPU内存未释放时的处理方法
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


保存所有日志
- 添加 `&> train.log 2>&1 &` 或 `&> train.log 2>&1`
